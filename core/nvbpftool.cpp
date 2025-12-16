/*
 * Example User Tool using NV-BPF
 * With Safety Features Enabled
 */

#include "nvbpf.h"

// --------------------------------------------------------------------
// 1. Define Maps
// --------------------------------------------------------------------

// Map Size: 128 SMs. 
// Uses SafeBpfMap automatically via macro.
BPF_MAP_DEF(sm_counter, uint64_t, 128);

// --------------------------------------------------------------------
// 2. Define Hooks
// --------------------------------------------------------------------

// Hook: Runs at the start of every GPU Kernel
SEC_KPROBE(track_kernel_execution) 
{
    // 1. Get Context
    NvBpfContext ctx;
    bpf_get_context(&ctx);

    int sm_id = ctx.sm_id;
    
    // --- SAFETY DEMO 1: Bounds Checking ---
    // Even if sm_id is somehow garbage (e.g. > 128), this will NOT crash.
    // It will silently increment the 'dropped_accesses' counter.
    sm_counter.atomic_inc(sm_id);

    // --- SAFETY DEMO 2: Safe Probe Read ---
    // Suppose we want to read a global value safely
    // (Simulating reading a kernel argument)
    // Here we just test reading from our own map safely
    uint64_t* map_val_ptr = sm_counter.lookup(sm_id);
    uint64_t local_val = 0;
    
    if (map_val_ptr) {
        // bpf_probe_read ensures alignment and null checks
        int ret = bpf_probe_read(&local_val, (const uint64_t*)map_val_ptr);
        
        // Only print if read was safe and successful
        if (ret == 0 && ctx.tid_x == 0 && ctx.cta_id_x == 0 && local_val % 100 == 0) {
            bpf_printk("SM %d has run %llu kernels (Safe Read)", sm_id, local_val);
        }
    }
    
    // --- SAFETY DEMO 3: Intentional Bad Access ---
    // If we tried this in raw CUDA, we might corrupt memory:
    // raw_array[99999] = 1; 
    // But with our wrapper, we can try accessing an invalid index:
    sm_counter.update(9999, 123); // This will simply fail safely
}

// --------------------------------------------------------------------
// 3. Boilerplate
// --------------------------------------------------------------------

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    printf("[NVBPF] Safe Tool Loaded.\n");
}

void nvbit_at_instruction(CUcontext ctx, CUfunction func) {
    nvbpf_attach_hooks(ctx, func);
}

// --------------------------------------------------------------------
// 4. Userspace Reader
// --------------------------------------------------------------------

extern "C" void report_results() {
    printf("\n--- NVBPF Safety Report ---\n");
    
    // Check for safety violations caught by the runtime
    if (sm_counter.dropped_accesses > 0) {
        printf("WARNING: Caught %llu unsafe/out-of-bounds memory accesses!\n", 
               sm_counter.dropped_accesses);
        printf("The GPU was saved from crashing.\n\n");
    } else {
        printf("No safety violations detected.\n");
    }

    printf("--- SM Execution Counts ---\n");
    for(int i=0; i<128; i++) {
        uint64_t val = *sm_counter.lookup(i);
        if(val > 0) {
            printf("SM %d: %lu executions\n", i, val);
        }
    }
}