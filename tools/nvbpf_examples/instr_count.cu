/*
 * NV-BPF Example: Instruction Counter
 *
 * A minimal example showing how to count GPU instructions
 * using the NV-BPF eBPF-style interface.
 *
 * Usage:
 *   make
 *   LD_PRELOAD=./instr_count.so ./your_cuda_app
 */

#include <pthread.h>
#include <unordered_set>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

/* ============================================
 * 1. Define Maps
 * ============================================ */

// Global instruction counter
BPF_ARRAY(instr_counter, uint64_t, 1);

// Per-SM counters (reduces contention)
BPF_PERCPU_ARRAY(sm_counters, uint64_t, 1);

/* ============================================
 * 2. Instrumentation
 * ============================================ */

// Device function is defined in instr_count_hooks.cu
extern "C" __device__ __noinline__ void count_instrs(int predicate,
                                                     uint64_t pcounter,
                                                     uint64_t psm_counters);

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
        for (auto* instr : instrs) {
            nvbit_insert_call(instr, "count_instrs", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val64(instr,
                                           (uint64_t)&instr_counter.data[0]);
            nvbit_add_call_arg_const_val64(instr,
                                           (uint64_t)&sm_counters.data[0][0]);
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
    printf("[NVBPF INSTR_COUNT] Tool loaded\n");
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (!nvbpf_is_launch_event(cbid)) return;

    CUfunction func = nvbpf_get_launch_func(cbid, params);

    if (!is_exit) {
        pthread_mutex_lock(&launch_mutex);
        instrument_function_if_needed(ctx, func);
        instr_counter.reset();
        sm_counters.reset();
        nvbit_enable_instrumented(ctx, func, true);
    } else {
        cudaDeviceSynchronize();

        uint64_t count = *instr_counter.lookup(0);
        printf("[NVBPF] Kernel %lu: %s - %lu warp instructions\n",
               kernel_count++,
               nvbit_get_func_name(ctx, func),
               count);
        pthread_mutex_unlock(&launch_mutex);
    }
}

void nvbit_at_term() {
    printf("\n[NVBPF INSTR_COUNT] === SM Distribution ===\n");

    for (int sm = 0; sm < 128; sm++) {
        uint64_t* val = sm_counters.lookup_sm(sm, 0);
        if (val && *val > 0) {
            printf("  SM %3d: %lu instructions\n", sm, *val);
        }
    }
    printf("[NVBPF INSTR_COUNT] Tool terminated\n");
}
