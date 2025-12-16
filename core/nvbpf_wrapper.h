#pragma once
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"

/* * ==========================================
 * NV-BPF: eBPF-style Wrapper for NVBit
 * with SAFETY RAILS enabled
 * ==========================================
 */

// --- 0. Safety Primitives ---

// Compile-time check: Ensure users don't use massive types in maps
template <typename T>
__device__ __host__ void bpf_static_check() {
    static_assert(sizeof(T) <= 256, "NVBPF Error: Map element size too large. Risk of cache thrashing.");
}

// Runtime Safety: Helper to check pointer alignment
__device__ inline bool bpf_is_aligned(const void* ptr, size_t align) {
    return ((uintptr_t)ptr % align) == 0;
}

// --- 1. Map Definitions (Safe Unified Memory) ---

template <typename T, int SIZE>
struct SafeBpfMap {
    T data[SIZE];
    
    // Debug counter for dropped/unsafe writes (visible to host)
    uint64_t dropped_accesses;

    SafeBpfMap() : dropped_accesses(0) {}

    // Safe Lookup: Returns nullptr if out of bounds
    __host__ __device__ T* lookup(int index) {
        if (index < 0 || index >= SIZE) {
            return nullptr; 
        }
        return &data[index];
    }

    // Safe Atomic Increment: Silently drops out-of-bounds writes
    __device__ void atomic_inc(int index) {
        if (index >= 0 && index < SIZE) {
            // Static check: Only allow atomics on supported types
            static_assert(sizeof(T) == 4 || sizeof(T) == 8, "Atomic increment only supports 32/64-bit integers");
            atomicAdd((unsigned long long*)&data[index], 1ULL);
        } else {
            // Log the failure safely
            atomicAdd((unsigned long long*)&dropped_accesses, 1ULL);
        }
    }

    // Safe Update: Returns 0 on success, -1 on error
    __device__ int update(int index, T value) {
        if (index < 0 || index >= SIZE) {
            atomicAdd((unsigned long long*)&dropped_accesses, 1ULL);
            return -1;
        }
        data[index] = value;
        return 0;
    }
};

// Macro to define a map easily
#define BPF_MAP_DEF(name, type, size) \
    __managed__ SafeBpfMap<type, size> name;

// --- 2. Context & Safe Helpers ---

struct NvBpfContext {
    int cta_id_x, cta_id_y, cta_id_z;
    int tid_x, tid_y, tid_z;
    int sm_id;
    uint64_t clock;
};

__device__ inline void bpf_get_context(NvBpfContext* ctx) {
    ctx->cta_id_x = blockIdx.x;
    ctx->cta_id_y = blockIdx.y;
    ctx->cta_id_z = blockIdx.z;
    ctx->tid_x = threadIdx.x;
    ctx->tid_y = threadIdx.y;
    ctx->tid_z = threadIdx.z;
    
    unsigned int smid;
    asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
    ctx->sm_id = smid;
    
    ctx->clock = clock64();
}

// Safe Memory Reader (Mimics bpf_probe_read)
// Prevents unaligned accesses which crash GPUs
template <typename T>
__device__ int bpf_probe_read(T* dst, const T* src) {
    if (src == nullptr || dst == nullptr) return -1;
    
    // Check alignment
    if (!bpf_is_aligned(src, alignof(T))) {
        return -2; // Error: Unaligned access
    }

    // Use __ldg (Load Global Cached) for read-only efficient access
    // Note: This assumes src points to global memory. 
    // For full safety, we'd need is_global_pointer() check but that's expensive.
    *dst = __ldg(src);
    return 0;
}

#define bpf_printk(fmt, ...) printf("[NVBPF] " fmt "\n", ##__VA_ARGS__)


// --- 3. Hook Registry System (Unchanged) ---

typedef void (*instrumentation_func_t)(int, int);

struct HookRegistry {
    const char* section_name;
    void* device_func; 
    bool is_registered;
};

HookRegistry kprobe_registry = { "kprobe", nullptr, false };

#define SEC_KPROBE(name) \
    __device__ void __nvbpf_##name(); \
    __global__ void __trampoline_##name(int n_args) { \
        __nvbpf_##name(); \
    } \
    struct __registrar_##name { \
        __registrar_##name() { \
            kprobe_registry.device_func = (void*)__trampoline_##name; \
            kprobe_registry.is_registered = true; \
        } \
    }; \
    static __registrar_##name __reg_inst_##name; \
    __device__ void __nvbpf_##name()


// --- 4. NVBit Backend (Unchanged Logic) ---

void nvbpf_attach_hooks(CUcontext ctx, CUfunction func) {
    const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, func);

    if (kprobe_registry.is_registered) {
        nvbit_insert_call(instrs[0], "eBPF_KProbe", (void*)kprobe_registry.device_func);
        // printf("[NVBPF] Attached KPROBE\n"); // Reduced noise
    }
}