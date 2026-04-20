/*
 * NV-BPF Example: Tail Fragment Tracker - Device Hooks
 */

#include <stdint.h>
#include "nvbpf_helpers.h"
#include "utils/utils.h"

static constexpr int SK_LOAD = 0;
static constexpr int SK_STORE = 1;
static constexpr int SK_MATH = 2;
static constexpr int SK_BRANCH = 3;
static constexpr int SK_ALL = 4;

static __device__ __forceinline__ void add_counter(uint64_t* values, int kind,
                                                   unsigned long long amount) {
    atomicAdd((unsigned long long*)&values[kind], amount);
    atomicAdd((unsigned long long*)&values[SK_ALL], amount);
}

extern "C" __device__ __noinline__ void tf_trace_site(int pred,
                                                      uint64_t ptotal,
                                                      uint64_t ppartial,
                                                      uint64_t pdead,
                                                      uint64_t plow,
                                                      uint64_t pwaste,
                                                      uint64_t pactive_sum,
                                                      uint64_t pwarp_sum,
                                                      uint64_t phist,
                                                      uint32_t threshold,
                                                      uint32_t kind) {
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), pred);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    const int warp_threads = __popc(active_mask);
    const int active_threads = __popc(predicate_mask);
    if (first_laneid != laneid) return;

    uint64_t* total = (uint64_t*)ptotal;
    uint64_t* partial = (uint64_t*)ppartial;
    uint64_t* dead = (uint64_t*)pdead;
    uint64_t* low = (uint64_t*)plow;
    uint64_t* waste = (uint64_t*)pwaste;
    uint64_t* active_sum = (uint64_t*)pactive_sum;
    uint64_t* warp_sum = (uint64_t*)pwarp_sum;
    uint64_t* hist = (uint64_t*)phist;

    add_counter(total, (int)kind, 1ULL);
    add_counter(warp_sum, (int)kind, (unsigned long long)warp_threads);
    if (active_threads >= 0 && active_threads <= 32) {
        atomicAdd((unsigned long long*)&hist[active_threads], 1ULL);
    }
    if ((uint32_t)active_threads < threshold) {
        add_counter(low, (int)kind, 1ULL);
    }
    if (active_threads == 0) {
        add_counter(dead, (int)kind, 1ULL);
        add_counter(waste, (int)kind, (unsigned long long)warp_threads);
        return;
    }

    add_counter(active_sum, (int)kind, (unsigned long long)active_threads);
    if (active_threads < warp_threads) {
        add_counter(partial, (int)kind, 1ULL);
        add_counter(waste, (int)kind,
                    (unsigned long long)(warp_threads - active_threads));
    }
}
