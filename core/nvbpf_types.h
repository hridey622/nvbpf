/*
 * NV-BPF: eBPF-style Wrapper for NVBit
 * Core Types and Context Structures
 * 
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once
#include <stdint.h>
#include <stdbool.h>

/* ============================================
 * Section 1: Compile-time Safety Primitives
 * ============================================ */

// Ensure map element sizes are cache-friendly
template <typename T>
struct BpfTypeCheck {
    static_assert(sizeof(T) <= 256, 
        "NVBPF Error: Type too large (max 256 bytes). Risk of cache thrashing.");
    static_assert(alignof(T) <= 16,
        "NVBPF Error: Type alignment too strict (max 16 bytes).");
};

/* ============================================
 * Section 2: Context Structures
 * ============================================ */

/**
 * GPU execution context - passed to all probes
 * Similar to pt_regs in Linux eBPF
 */
struct NvBpfContext {
    // Thread/Block identification (always available)
    int cta_id_x, cta_id_y, cta_id_z;   // Block index
    int tid_x, tid_y, tid_z;             // Thread index within block
    int warp_id;                         // Warp ID within block
    int lane_id;                         // Lane within warp (0-31)
    int sm_id;                           // Streaming Multiprocessor ID
    
    // Grid dimensions
    int grid_dim_x, grid_dim_y, grid_dim_z;
    int block_dim_x, block_dim_y, block_dim_z;
    
    // Timing
    uint64_t clock;                      // GPU clock cycles
    
    // Memory access info (for memory tracepoints)
    uint64_t addr;                       // Memory address being accessed
    uint32_t size;                       // Access size in bytes
    uint32_t opcode_id;                  // Opcode identifier
    uint32_t instr_offset;               // Instruction offset in function
    
    // Flags
    uint8_t is_load : 1;
    uint8_t is_store : 1;
    uint8_t is_atomic : 1;
    uint8_t predicate : 1;               // Instruction predicate value
    uint8_t _reserved : 4;
};

/* ============================================
 * Section 3: Hook/Probe Types
 * ============================================ */

enum NvBpfHookType {
    HOOK_NONE = 0,
    
    // Kernel-level probes
    HOOK_KPROBE_ENTRY,          // First instruction of kernel
    HOOK_KRETPROBE_EXIT,        // RET instructions
    
    // Memory tracepoints  
    HOOK_TRACEPOINT_MEM_LOAD,   // Load instructions
    HOOK_TRACEPOINT_MEM_STORE,  // Store instructions
    HOOK_TRACEPOINT_MEM_ATOMIC, // Atomic operations
    
    // Instruction tracepoints
    HOOK_TRACEPOINT_INSTR_ALL,  // Every instruction
    HOOK_TRACEPOINT_INSTR_BRANCH, // Branch instructions
    HOOK_TRACEPOINT_INSTR_OPCODE, // Specific opcode
    
    HOOK_TYPE_COUNT
};

/**
 * Hook registration entry
 */
struct NvBpfHookEntry {
    NvBpfHookType type;
    const char* name;              // Human-readable name
    const char* device_func_name;  // __device__ function to call
    const char* filter;            // Optional: opcode filter pattern
    bool enabled;
};

/* ============================================
 * Section 4: Map Types
 * ============================================ */

enum NvBpfMapType {
    MAP_TYPE_ARRAY,
    MAP_TYPE_HASH,
    MAP_TYPE_PERCPU_ARRAY,
    MAP_TYPE_RINGBUF
};

/* ============================================
 * Section 5: Error Codes
 * ============================================ */

#define NVBPF_OK              0
#define NVBPF_ERR_NOT_FOUND  -1
#define NVBPF_ERR_NO_SPACE   -2
#define NVBPF_ERR_INVALID    -3
#define NVBPF_ERR_UNALIGNED  -4
#define NVBPF_ERR_NULLPTR    -5

/* ============================================
 * Section 6: Utility Macros
 * ============================================ */

// Stringify helper
#define NVBPF_STRINGIFY(x) #x
#define NVBPF_TOSTRING(x) NVBPF_STRINGIFY(x)

// Unique name generator
#define NVBPF_CONCAT_(a, b) a##b
#define NVBPF_CONCAT(a, b) NVBPF_CONCAT_(a, b)
#define NVBPF_UNIQUE(prefix) NVBPF_CONCAT(prefix, __COUNTER__)
