/*
 * NV-BPF: eBPF-style Wrapper for NVBit
 * Helper Functions
 * 
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once
#include <stdint.h>
#include <stdio.h>
#include "nvbpf_types.h"

/* ============================================
 * Section 1: Context Helpers
 * ============================================ */

/**
 * Get current SM ID
 */
__device__ __forceinline__ uint32_t bpf_get_current_sm_id() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

/**
 * Get current warp ID within the block
 */
__device__ __forceinline__ uint32_t bpf_get_current_warp_id() {
    return threadIdx.x / 32 + 
           (threadIdx.y * blockDim.x / 32) + 
           (threadIdx.z * blockDim.x * blockDim.y / 32);
}

/**
 * Get lane ID within the warp (0-31)
 */
__device__ __forceinline__ uint32_t bpf_get_current_lane_id() {
    uint32_t laneid;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
    return laneid;
}

/**
 * Get GPU nanosecond timestamp (approximate)
 * Note: Actual resolution depends on GPU clock frequency
 */
__device__ __forceinline__ uint64_t bpf_ktime_get_ns() {
    return clock64();
}

/**
 * Get global thread ID (linearized)
 */
__device__ __forceinline__ uint64_t bpf_get_global_thread_id() {
    uint64_t tid = threadIdx.x + threadIdx.y * blockDim.x + 
                   threadIdx.z * blockDim.x * blockDim.y;
    uint64_t bid = blockIdx.x + blockIdx.y * gridDim.x + 
                   blockIdx.z * gridDim.x * gridDim.y;
    uint64_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    return bid * threads_per_block + tid;
}

/**
 * Fill NvBpfContext with current execution state
 */
__device__ __forceinline__ void bpf_get_context(NvBpfContext* ctx) {
    // Block/CTA IDs
    ctx->cta_id_x = blockIdx.x;
    ctx->cta_id_y = blockIdx.y;
    ctx->cta_id_z = blockIdx.z;
    
    // Thread IDs
    ctx->tid_x = threadIdx.x;
    ctx->tid_y = threadIdx.y;
    ctx->tid_z = threadIdx.z;
    
    // Warp/Lane
    ctx->warp_id = bpf_get_current_warp_id();
    ctx->lane_id = bpf_get_current_lane_id();
    ctx->sm_id = bpf_get_current_sm_id();
    
    // Dimensions
    ctx->grid_dim_x = gridDim.x;
    ctx->grid_dim_y = gridDim.y;
    ctx->grid_dim_z = gridDim.z;
    ctx->block_dim_x = blockDim.x;
    ctx->block_dim_y = blockDim.y;
    ctx->block_dim_z = blockDim.z;
    
    // Timing
    ctx->clock = clock64();
    
    // Memory fields default to 0 (filled by tracepoint hooks)
    ctx->addr = 0;
    ctx->size = 0;
    ctx->opcode_id = 0;
    ctx->instr_offset = 0;
    ctx->is_load = 0;
    ctx->is_store = 0;
    ctx->is_atomic = 0;
    ctx->predicate = 1;
}

/* ============================================
 * Section 2: Memory Safety Helpers
 * ============================================ */

/**
 * Check pointer alignment
 */
__device__ __forceinline__ bool bpf_is_aligned(const void* ptr, size_t align) {
    return ((uintptr_t)ptr % align) == 0;
}

/**
 * Safe memory read (mimics bpf_probe_read_kernel)
 * Returns: 0 on success, negative error code on failure
 */
template <typename T>
__device__ __forceinline__ int bpf_probe_read_kernel(T* dst, const T* src) {
    if (src == nullptr || dst == nullptr) {
        return NVBPF_ERR_NULLPTR;
    }
    
    if (!bpf_is_aligned(src, alignof(T))) {
        return NVBPF_ERR_UNALIGNED;
    }
    
    // Use cached load for efficiency
    *dst = __ldg(src);
    return NVBPF_OK;
}

/**
 * Safe memory read with size parameter
 */
__device__ __forceinline__ int bpf_probe_read_kernel_bytes(
    void* dst, const void* src, uint32_t size) 
{
    if (src == nullptr || dst == nullptr) {
        return NVBPF_ERR_NULLPTR;
    }
    
    // Byte-by-byte copy for safety
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* d = (uint8_t*)dst;
    for (uint32_t i = 0; i < size; i++) {
        d[i] = __ldg(&s[i]);
    }
    return NVBPF_OK;
}

/* ============================================
 * Section 3: Atomic Helpers
 * ============================================ */

/**
 * Atomic increment (64-bit)
 */
__device__ __forceinline__ uint64_t bpf_atomic_inc(uint64_t* ptr) {
    return atomicAdd((unsigned long long*)ptr, 1ULL);
}

/**
 * Atomic add (64-bit)
 */
__device__ __forceinline__ uint64_t bpf_atomic_add(uint64_t* ptr, uint64_t val) {
    return atomicAdd((unsigned long long*)ptr, val);
}

/**
 * Atomic compare-and-swap (64-bit)
 */
__device__ __forceinline__ uint64_t bpf_atomic_cmpxchg(
    uint64_t* ptr, uint64_t expected, uint64_t desired) 
{
    return atomicCAS((unsigned long long*)ptr, expected, desired);
}

/* ============================================
 * Section 4: Warp-level Helpers
 * ============================================ */

/**
 * Count active threads in warp
 */
__device__ __forceinline__ uint32_t bpf_warp_active_count() {
    return __popc(__activemask());
}

/**
 * Check if this is the first active lane
 */
__device__ __forceinline__ bool bpf_is_warp_leader() {
    uint32_t mask = __activemask();
    uint32_t leader = __ffs(mask) - 1;
    return bpf_get_current_lane_id() == leader;
}

/* ============================================
 * Section 5: Debug Helpers
 * ============================================ */

/**
 * Printf wrapper with NVBPF prefix
 */
#define bpf_printk(fmt, ...) \
    printf("[NVBPF] " fmt "\n", ##__VA_ARGS__)

/**
 * Conditional print (only first thread in warp)
 */
#define bpf_printk_once(fmt, ...) \
    do { \
        if (bpf_is_warp_leader()) { \
            printf("[NVBPF] " fmt "\n", ##__VA_ARGS__); \
        } \
    } while(0)

/**
 * Rate-limited print (every N calls)
 */
#define bpf_printk_ratelimit(counter_ptr, n, fmt, ...) \
    do { \
        if (atomicAdd((unsigned long long*)(counter_ptr), 1ULL) % (n) == 0) { \
            printf("[NVBPF] " fmt "\n", ##__VA_ARGS__); \
        } \
    } while(0)
