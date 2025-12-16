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
 * 2. Define Hooks
 * ============================================ */

// Run at kernel entry to each thread
SEC_KPROBE(kernel_entry) {
    BPF_REQUIRE_PRED(pred);
    BPF_WARP_LEADER_ONLY();
    
    // Count entries per SM
    sm_entries.atomic_inc(0);
    
    // Mark SM as active in bitmap
    uint32_t sm = bpf_get_current_sm_id();
    uint32_t word = sm / 64;
    uint64_t bit = 1ULL << (sm % 64);
    
    if (word < 4) {
        uint64_t* bm = active_sm_bitmap.lookup(word);
        if (bm) {
            atomicOr((unsigned long long*)bm, bit);
        }
    }
}

// Count instructions per SM
SEC_TRACEPOINT_INSTR(count_per_sm) {
    BPF_REQUIRE_PRED(pred);
    BPF_WARP_LEADER_ONLY();
    
    sm_instrs.atomic_inc(0);
}

/* ============================================
 * 3. NVBit Callbacks
 * ============================================ */

static uint64_t kernel_count = 0;

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    printf("[NVBPF SM_PROFILER] Tool loaded\n");
    nvbpf::nvbpf_print_hooks();
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (!nvbpf_is_launch_event(cbid)) return;
    
    CUfunction func = nvbpf_get_launch_func(cbid, params);
    
    if (!is_exit) {
        nvbpf_attach_hooks(ctx, func);
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
        
        // Reset
        sm_instrs.reset();
        sm_entries.reset();
        active_sm_bitmap.reset();
    }
}

void nvbit_at_term() {
    printf("[NVBPF SM_PROFILER] Tool terminated\n");
}

void nvbit_at_ctx_init(CUcontext ctx) {}
void nvbit_at_ctx_term(CUcontext ctx) {}
