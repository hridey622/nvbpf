/*
 * NV-BPF Example: GEMM Wavefit Trace - Device Hooks
 */

#include <stdint.h>
#include "nvbpf_helpers.h"

static constexpr uint32_t kNvbpfMaxSms = 256;
static constexpr uint32_t kBitmapWords = 4;

extern "C" __device__ __noinline__ void gwt_kernel_entry(int pred,
                                                         uint64_t psm_entries,
                                                         uint64_t pbitmap) {
    if (!pred) return;
    if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) return;

    uint32_t sm = bpf_get_current_sm_id();
    if (sm >= kNvbpfMaxSms) return;

    uint64_t* sm_entries = (uint64_t*)psm_entries;
    atomicAdd((unsigned long long*)&sm_entries[sm], 1ULL);

    uint32_t word = sm / 64;
    if (word < kBitmapWords) {
        uint64_t* bitmap = (uint64_t*)pbitmap;
        atomicOr((unsigned long long*)&bitmap[word], 1ULL << (sm % 64));
    }
}

