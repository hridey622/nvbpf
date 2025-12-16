/*
 * NV-BPF: eBPF-style Wrapper for NVBit
 * Hook/Probe Registration System (SEC macros)
 * 
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once
#include <stdint.h>
#include <vector>
#include <cstring>
#include "nvbpf_types.h"

/* ============================================
 * Section 1: Hook Registry
 * ============================================ */

#define NVBPF_MAX_HOOKS 64

/**
 * Global hook registry - tracks all registered probes
 */
struct NvBpfHookRegistry {
    NvBpfHookEntry entries[NVBPF_MAX_HOOKS];
    int count;
    
    NvBpfHookRegistry() : count(0) {
        for (int i = 0; i < NVBPF_MAX_HOOKS; i++) {
            entries[i].type = HOOK_NONE;
            entries[i].enabled = false;
        }
    }
    
    int add(NvBpfHookType type, const char* name, const char* func_name, 
            const char* filter = nullptr) {
        if (count >= NVBPF_MAX_HOOKS) return -1;
        
        entries[count].type = type;
        entries[count].name = name;
        entries[count].device_func_name = func_name;
        entries[count].filter = filter;
        entries[count].enabled = true;
        return count++;
    }
    
    NvBpfHookEntry* find_by_type(NvBpfHookType type) {
        for (int i = 0; i < count; i++) {
            if (entries[i].type == type && entries[i].enabled) {
                return &entries[i];
            }
        }
        return nullptr;
    }
    
    template <typename Callback>
    void for_each(Callback&& cb) {
        for (int i = 0; i < count; i++) {
            if (entries[i].enabled) {
                cb(&entries[i]);
            }
        }
    }
};

// Global registry instance
inline NvBpfHookRegistry& nvbpf_get_hook_registry() {
    static NvBpfHookRegistry registry;
    return registry;
}

/* ============================================
 * Section 2: Hook Registration Helpers
 * ============================================ */

/**
 * RAII helper for auto-registering hooks at static init time
 */
struct NvBpfHookRegistrar {
    NvBpfHookRegistrar(NvBpfHookType type, const char* name, 
                       const char* func_name, const char* filter = nullptr) {
        nvbpf_get_hook_registry().add(type, name, func_name, filter);
    }
};

/* ============================================
 * Section 3: SEC Macros - Kernel Entry/Exit
 * ============================================ */

/**
 * SEC("kprobe/kernel_entry") - Runs at kernel start
 * 
 * Usage:
 *   SEC_KPROBE(my_probe_name) {
 *       NvBpfContext ctx;
 *       bpf_get_context(&ctx);
 *       // ... instrumentation code
 *   }
 */
#define SEC_KPROBE(name) \
    extern "C" __device__ __noinline__ void __nvbpf_kprobe_##name(int pred); \
    static NvBpfHookRegistrar __nvbpf_reg_kprobe_##name( \
        HOOK_KPROBE_ENTRY, #name, "__nvbpf_kprobe_" #name); \
    extern "C" __device__ __noinline__ void __nvbpf_kprobe_##name(int pred)

/**
 * SEC("kretprobe/kernel_exit") - Runs at kernel return
 */
#define SEC_KRETPROBE(name) \
    extern "C" __device__ __noinline__ void __nvbpf_kretprobe_##name(int pred); \
    static NvBpfHookRegistrar __nvbpf_reg_kretprobe_##name( \
        HOOK_KRETPROBE_EXIT, #name, "__nvbpf_kretprobe_" #name); \
    extern "C" __device__ __noinline__ void __nvbpf_kretprobe_##name(int pred)

/* ============================================
 * Section 4: SEC Macros - Memory Tracepoints
 * ============================================ */

/**
 * SEC("tracepoint/memory/load") - Runs on memory loads
 * 
 * The probe receives:
 *   - pred: predicate value (1 if instruction executes)
 *   - addr: 64-bit memory address being loaded
 * 
 * Usage:
 *   SEC_TRACEPOINT_MEM_LOAD(my_load_trace) {
 *       if (!pred) return;
 *       // addr is available as parameter
 *   }
 */
#define SEC_TRACEPOINT_MEM_LOAD(name) \
    extern "C" __device__ __noinline__ void __nvbpf_tp_mem_load_##name( \
        int pred, uint64_t addr); \
    static NvBpfHookRegistrar __nvbpf_reg_tp_mem_load_##name( \
        HOOK_TRACEPOINT_MEM_LOAD, #name, "__nvbpf_tp_mem_load_" #name); \
    extern "C" __device__ __noinline__ void __nvbpf_tp_mem_load_##name( \
        int pred, uint64_t addr)

/**
 * SEC("tracepoint/memory/store") - Runs on memory stores
 */
#define SEC_TRACEPOINT_MEM_STORE(name) \
    extern "C" __device__ __noinline__ void __nvbpf_tp_mem_store_##name( \
        int pred, uint64_t addr); \
    static NvBpfHookRegistrar __nvbpf_reg_tp_mem_store_##name( \
        HOOK_TRACEPOINT_MEM_STORE, #name, "__nvbpf_tp_mem_store_" #name); \
    extern "C" __device__ __noinline__ void __nvbpf_tp_mem_store_##name( \
        int pred, uint64_t addr)

/**
 * SEC("tracepoint/memory/all") - Runs on any memory operation
 */
#define SEC_TRACEPOINT_MEM(name) \
    extern "C" __device__ __noinline__ void __nvbpf_tp_mem_##name( \
        int pred, uint64_t addr, int is_load, int is_store); \
    static NvBpfHookRegistrar __nvbpf_reg_tp_mem_load_##name( \
        HOOK_TRACEPOINT_MEM_LOAD, #name "_load", "__nvbpf_tp_mem_" #name); \
    static NvBpfHookRegistrar __nvbpf_reg_tp_mem_store_##name( \
        HOOK_TRACEPOINT_MEM_STORE, #name "_store", "__nvbpf_tp_mem_" #name); \
    extern "C" __device__ __noinline__ void __nvbpf_tp_mem_##name( \
        int pred, uint64_t addr, int is_load, int is_store)

/* ============================================
 * Section 5: SEC Macros - Instruction Tracepoints
 * ============================================ */

/**
 * SEC("tracepoint/instruction/all") - Runs on every instruction
 * 
 * Parameters:
 *   - pred: predicate value
 *   - opcode_id: unique ID for this opcode
 */
#define SEC_TRACEPOINT_INSTR(name) \
    extern "C" __device__ __noinline__ void __nvbpf_tp_instr_##name( \
        int pred, uint32_t opcode_id); \
    static NvBpfHookRegistrar __nvbpf_reg_tp_instr_##name( \
        HOOK_TRACEPOINT_INSTR_ALL, #name, "__nvbpf_tp_instr_" #name); \
    extern "C" __device__ __noinline__ void __nvbpf_tp_instr_##name( \
        int pred, uint32_t opcode_id)

/**
 * SEC("tracepoint/instruction/branch") - Runs on branch instructions
 */
#define SEC_TRACEPOINT_BRANCH(name) \
    extern "C" __device__ __noinline__ void __nvbpf_tp_branch_##name(int pred); \
    static NvBpfHookRegistrar __nvbpf_reg_tp_branch_##name( \
        HOOK_TRACEPOINT_INSTR_BRANCH, #name, "__nvbpf_tp_branch_" #name); \
    extern "C" __device__ __noinline__ void __nvbpf_tp_branch_##name(int pred)

/**
 * SEC("tracepoint/instruction/OPCODE") - Runs on specific opcode
 * 
 * Usage:
 *   SEC_TRACEPOINT_OPCODE(count_fma, "FFMA") { ... }
 */
#define SEC_TRACEPOINT_OPCODE(name, opcode_filter) \
    extern "C" __device__ __noinline__ void __nvbpf_tp_opcode_##name(int pred); \
    static NvBpfHookRegistrar __nvbpf_reg_tp_opcode_##name( \
        HOOK_TRACEPOINT_INSTR_OPCODE, #name, "__nvbpf_tp_opcode_" #name, opcode_filter); \
    extern "C" __device__ __noinline__ void __nvbpf_tp_opcode_##name(int pred)

/* ============================================
 * Section 6: Utility Macros
 * ============================================ */

/**
 * Early return if predicate is false
 * Use at start of probe to skip predicated-off instructions
 */
#define BPF_REQUIRE_PRED(pred) \
    do { if (!(pred)) return; } while(0)

/**
 * Execute only for one thread per warp
 * Useful for warp-level counting
 */
#define BPF_WARP_LEADER_ONLY() \
    do { if (!bpf_is_warp_leader()) return; } while(0)

/**
 * Execute only for thread (0,0,0) in block
 */
#define BPF_BLOCK_LEADER_ONLY() \
    do { if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) return; } while(0)
