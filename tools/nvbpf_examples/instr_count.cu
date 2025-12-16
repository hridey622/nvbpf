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
 * 2. Define Hooks
 * ============================================ */

// Count every instruction (warp-level)
SEC_TRACEPOINT_INSTR(count_all) {
    BPF_REQUIRE_PRED(pred);
    BPF_WARP_LEADER_ONLY();
    
    // Increment global counter
    instr_counter.atomic_inc(0);
    
    // Also increment per-SM counter
    sm_counters.atomic_inc(0);
}

/* ============================================
 * 3. NVBit Callbacks
 * ============================================ */

static uint64_t kernel_count = 0;

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    printf("[NVBPF INSTR_COUNT] Tool loaded\n");
    nvbpf::nvbpf_print_hooks();
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (!nvbpf_is_launch_event(cbid)) return;
    
    CUfunction func = nvbpf_get_launch_func(cbid, params);
    
    if (!is_exit) {
        // Before kernel: attach hooks and reset counter
        nvbpf_attach_hooks(ctx, func);
    } else {
        // After kernel: print results
        cudaDeviceSynchronize();
        
        uint64_t count = *instr_counter.lookup(0);
        printf("[NVBPF] Kernel %lu: %s - %lu warp instructions\n",
               kernel_count++, 
               nvbit_get_func_name(ctx, func),
               count);
        
        // Reset for next kernel
        instr_counter.reset();
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

// Unused callbacks
void nvbit_at_ctx_init(CUcontext ctx) {}
void nvbit_at_ctx_term(CUcontext ctx) {}
