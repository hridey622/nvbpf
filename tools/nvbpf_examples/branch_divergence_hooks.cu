/*
 * NV-BPF Example: Branch Divergence - Device Hooks
 */

#include <stdint.h>
#include "nvbpf_helpers.h"
#include "utils/utils.h"

static constexpr uint32_t kNvbpfMaxSms = 256;

extern "C" __device__ __noinline__ void bd_trace_branch(int pred,
                                                        uint64_t ptotal,
                                                        uint64_t pdivergent,
                                                        uint64_t ppred_off,
                                                        uint64_t pactive_sum,
                                                        uint64_t phist,
                                                        uint64_t pdivergent_sm,
                                                        uint32_t threshold) {
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), pred);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    const int active_threads = __popc(predicate_mask);
    if (first_laneid != laneid) return;

    atomicAdd((unsigned long long*)ptotal, 1ULL);
    uint64_t* hist = (uint64_t*)phist;
    if (active_threads >= 0 && active_threads <= 32) {
        atomicAdd((unsigned long long*)&hist[active_threads], 1ULL);
    }

    if (active_threads == 0) {
        atomicAdd((unsigned long long*)ppred_off, 1ULL);
        return;
    }

    atomicAdd((unsigned long long*)pactive_sum, (unsigned long long)active_threads);
    if ((uint32_t)active_threads < threshold) {
        atomicAdd((unsigned long long*)pdivergent, 1ULL);
        uint32_t sm = bpf_get_current_sm_id();
        if (sm < kNvbpfMaxSms) {
            uint64_t* per_sm = (uint64_t*)pdivergent_sm;
            atomicAdd((unsigned long long*)&per_sm[sm], 1ULL);
        }
    }
}
