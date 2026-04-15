/*
 * NV-BPF Example: SM Profiler
 * 
 * Profiles execution distribution across SMs using
 * per-CPU (per-SM) maps for low-contention counting.
 * 
 * Usage:
 *   make
 *   LD_PRELOAD=./sm_profiler.so ./your_cuda_app
 */

#include <pthread.h>
#include <unordered_set>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

/* ============================================
 * 1. Define Maps
 * ============================================ */

// Per-SM instruction counts (reduces atomic contention)
BPF_PERCPU_ARRAY(sm_instrs, uint64_t, 1);

// Per-SM kernel entry counts
BPF_PERCPU_ARRAY(sm_entries, uint64_t, 1);

// Track active SMs
BPF_ARRAY(active_sm_bitmap, uint64_t, 4); // 256 bits = 4 x 64

/* ============================================
 * 2. Instrumentation
 * ============================================ */

extern "C" __device__ __noinline__ void sm_kernel_entry(int pred,
                                                        uint64_t psm_entries,
                                                        uint64_t pbitmap);
extern "C" __device__ __noinline__ void sm_count_instr(int pred,
                                                       uint64_t psm_instrs);

static pthread_mutex_t launch_mutex;
static std::unordered_set<CUfunction> already_instrumented;

static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);

    for (auto f : related) {
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        if (instrs.empty()) {
            continue;
        }

        nvbit_insert_call(instrs[0], "sm_kernel_entry", IPOINT_BEFORE);
        nvbit_add_call_arg_guard_pred_val(instrs[0]);
        nvbit_add_call_arg_const_val64(instrs[0],
                                       (uint64_t)&sm_entries.data[0][0]);
        nvbit_add_call_arg_const_val64(instrs[0],
                                       (uint64_t)&active_sm_bitmap.data[0]);

        for (auto* instr : instrs) {
            nvbit_insert_call(instr, "sm_count_instr", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val64(instr,
                                           (uint64_t)&sm_instrs.data[0][0]);
        }
    }
}

/* ============================================
 * 3. NVBit Callbacks
 * ============================================ */

static uint64_t kernel_count = 0;

void nvbit_at_init() {
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    pthread_mutex_init(&launch_mutex, nullptr);
    printf("[NVBPF SM_PROFILER] Tool loaded\n");
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (!nvbpf_is_launch_event(cbid)) return;
    
    CUfunction func = nvbpf_get_launch_func(cbid, params);
    
    if (!is_exit) {
        pthread_mutex_lock(&launch_mutex);
        instrument_function_if_needed(ctx, func);
        sm_instrs.reset();
        sm_entries.reset();
        active_sm_bitmap.reset();
        nvbit_enable_instrumented(ctx, func, true);
    } else {
        cudaDeviceSynchronize();
        
        printf("\n[NVBPF] Kernel %lu: %s\n", kernel_count++,
               nvbit_get_func_name(ctx, func));
        
        // Find active SMs
        int active_count = 0;
        for (int word = 0; word < 4; word++) {
            uint64_t* bm = active_sm_bitmap.lookup(word);
            if (bm) active_count += __builtin_popcountll(*bm);
        }
        printf("        Active SMs: %d\n", active_count);
        
        // Print per-SM stats
        printf("        SM Distribution:\n");
        uint64_t total_instrs = 0;
        for (int sm = 0; sm < 128; sm++) {
            uint64_t* instrs = sm_instrs.lookup_sm(sm, 0);
            uint64_t* entries = sm_entries.lookup_sm(sm, 0);
            
            if (instrs && *instrs > 0) {
                printf("          SM %3d: %8lu instrs, %6lu entries\n",
                       sm, *instrs, entries ? *entries : 0);
                total_instrs += *instrs;
            }
        }
        printf("        Total: %lu warp instructions\n", total_instrs);
        
        pthread_mutex_unlock(&launch_mutex);
    }
}

void nvbit_at_term() {
    printf("[NVBPF SM_PROFILER] Tool terminated\n");
}
