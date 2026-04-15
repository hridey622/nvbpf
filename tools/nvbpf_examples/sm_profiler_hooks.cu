/*
 * NV-BPF Example: SM Profiler - Device Hooks
 * 
 * Device-side instrumentation functions for SM profiling.
 * Compiled with --keep-device-functions and -Xptxas -astoolspatch.
 */

#include <stdint.h>
#include "nvbpf_helpers.h"
#include "nvbpf_types.h"

static constexpr uint32_t kNvbpfMaxSms = 256;
static constexpr uint32_t kBitmapWords = 4;

extern "C" __device__ __noinline__ void sm_kernel_entry(int pred,
                                                        uint64_t psm_entries,
                                                        uint64_t pbitmap) {
    if (!pred) return;
    if (!bpf_is_warp_leader()) return;

    uint32_t sm = bpf_get_current_sm_id();
    if (sm >= kNvbpfMaxSms) return;

    uint64_t* sm_entries = (uint64_t*)psm_entries;
    atomicAdd((unsigned long long*)&sm_entries[sm], 1ULL);

    uint32_t word = sm / 64;
    uint64_t bit = 1ULL << (sm % 64);

    if (word < kBitmapWords) {
        uint64_t* bitmap = (uint64_t*)pbitmap;
        atomicOr((unsigned long long*)&bitmap[word], bit);
    }
}

extern "C" __device__ __noinline__ void sm_count_instr(int pred,
                                                       uint64_t psm_instrs) {
    if (!pred) return;
    if (!bpf_is_warp_leader()) return;

    uint32_t sm = bpf_get_current_sm_id();
    if (sm >= kNvbpfMaxSms) return;

    uint64_t* sm_instrs = (uint64_t*)psm_instrs;
    atomicAdd((unsigned long long*)&sm_instrs[sm], 1ULL);
}
