/*
 * NV-BPF: eBPF-style Wrapper for NVBit
 * Main Include Header
 * 
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 * ============================================
 * USAGE OVERVIEW
 * ============================================
 * 
 * NV-BPF provides eBPF-like abstractions for GPU instrumentation:
 * 
 * 1. MAPS - Shared memory between GPU instrumentation and CPU analysis:
 *    - BPF_ARRAY(name, type, size)     - Fixed-size array with bounds checking
 *    - BPF_HASH(name, K, V, size)      - Key-value hash map
 *    - BPF_PERCPU_ARRAY(name, type, n) - Per-SM private arrays
 *    - BPF_RINGBUF(name, type, size)   - GPU→CPU data streaming
 * 
 * 2. HOOKS/PROBES - Attach code to GPU execution events:
 *    - SEC_KPROBE(name)                - Kernel entry point
 *    - SEC_KRETPROBE(name)             - Kernel exit point
 *    - SEC_TRACEPOINT_MEM_LOAD(name)   - Memory load operations
 *    - SEC_TRACEPOINT_MEM_STORE(name)  - Memory store operations
 *    - SEC_TRACEPOINT_INSTR(name)      - Every instruction
 *    - SEC_TRACEPOINT_OPCODE(name, op) - Specific opcode
 * 
 * 3. HELPERS - Safe utility functions:
 *    - bpf_get_context(&ctx)           - Fill execution context
 *    - bpf_get_current_sm_id()         - Get SM ID
 *    - bpf_ktime_get_ns()              - GPU timestamp
 *    - bpf_probe_read_kernel(&dst,src) - Safe memory read
 *    - bpf_printk(fmt, ...)            - Debug output
 * 
 * EXAMPLE:
 * 
 *   #include "nvbpf.h"
 *   
 *   BPF_ARRAY(counter, uint64_t, 1);
 *   
 *   SEC_KPROBE(count_kernels) {
 *       BPF_REQUIRE_PRED(pred);
 *       counter.atomic_inc(0);
 *   }
 *   
 *   void nvbit_at_cuda_event(...) {
 *       if (!is_exit && is_launch_event(cbid)) {
 *           nvbpf_attach_hooks(ctx, func);
 *       }
 *   }
 * 
 * ============================================
 */

#pragma once

/* Core NVBit includes */
#include "nvbit_tool.h"
#include "nvbit.h"

/* NV-BPF components */
#include "nvbpf_types.h"    /* Types, context structures, enums */
#include "nvbpf_helpers.h"  /* Helper functions */
#include "nvbpf_maps.h"     /* Map definitions */
#include "nvbpf_hooks.h"    /* SEC macros and hook registry */
#include "nvbpf_loader.h"   /* Auto-injection backend */

/* Optional: Original wrapper for compatibility */
/* #include "nvbpf_wrapper.h" */

/* ============================================
 * Convenience Utilities
 * ============================================ */

/**
 * Check if a CUDA event is a kernel launch
 */
inline bool nvbpf_is_launch_event(nvbit_api_cuda_t cbid) {
    return (cbid == API_CUDA_cuLaunch ||
            cbid == API_CUDA_cuLaunchKernel ||
            cbid == API_CUDA_cuLaunchKernel_ptsz ||
            cbid == API_CUDA_cuLaunchGrid ||
            cbid == API_CUDA_cuLaunchGridAsync ||
            cbid == API_CUDA_cuLaunchKernelEx ||
            cbid == API_CUDA_cuLaunchKernelEx_ptsz);
}

/**
 * Extract CUfunction from launch event parameters
 */
inline CUfunction nvbpf_get_launch_func(nvbit_api_cuda_t cbid, void* params) {
    if (cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        return ((cuLaunchKernelEx_params*)params)->f;
    }
    // All other launch types use cuLaunchKernel_params layout
    return ((cuLaunchKernel_params*)params)->f;
}

/* ============================================
 * Default NVBit Callback Implementations
 * (Weak symbols - can be overridden)
 * ============================================ */

#ifndef NVBPF_NO_DEFAULT_CALLBACKS

/**
 * Default nvbit_at_init - sets up managed memory
 */
__attribute__((weak))
void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    printf("[NVBPF] NV-BPF Tool Loaded\n");
    nvbpf::nvbpf_print_hooks();
}

/**
 * Default nvbit_at_term - cleanup
 */
__attribute__((weak)) 
void nvbit_at_term() {
    printf("[NVBPF] NV-BPF Tool Terminated\n");
}

/**
 * Default context init/term - no-op
 */
__attribute__((weak))
void nvbit_at_ctx_init(CUcontext ctx) {}

__attribute__((weak))
void nvbit_at_ctx_term(CUcontext ctx) {}

__attribute__((weak))
void nvbit_tool_init(CUcontext ctx) {}

#endif /* NVBPF_NO_DEFAULT_CALLBACKS */
