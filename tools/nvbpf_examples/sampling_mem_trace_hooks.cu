/*
 * NV-BPF Example: Sampling Memory Trace - Device Hooks
 */

#include <stdint.h>
#include "nvbpf_helpers.h"
#include "nvbpf_maps.h"
#include "utils/utils.h"

struct SampledMemEvent {
    uint64_t addr;
    uint32_t sm_id;
    uint32_t warp_id;
    uint16_t cta_id_x;
    uint16_t cta_id_y;
    uint8_t is_load;
    uint8_t _pad[3];
};

static constexpr int kRingCapacity = 8192;

extern "C" __device__ __noinline__ void smt_trace_mem(int pred,
                                                      uint64_t addr,
                                                      uint64_t pring,
                                                      uint64_t ptotal,
                                                      uint64_t pkept,
                                                      uint32_t sample_every,
                                                      uint64_t addr_min,
                                                      uint64_t addr_max,
                                                      int is_load) {
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), pred);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    const int num_threads = __popc(predicate_mask);
    if (first_laneid != laneid || num_threads == 0) return;

    atomicAdd((unsigned long long*)ptotal, 1ULL);
    if (addr < addr_min || addr > addr_max) return;

    uint32_t sm = bpf_get_current_sm_id();
    uint32_t warp = bpf_get_current_warp_id();
    if (sample_every > 1) {
        uint64_t key = ((addr >> 5) ^ (uint64_t)sm ^ ((uint64_t)warp << 8));
        if ((key % sample_every) != 0) return;
    }

    atomicAdd((unsigned long long*)pkept, 1ULL);
    SampledMemEvent evt;
    evt.addr = addr;
    evt.sm_id = sm;
    evt.warp_id = warp;
    evt.cta_id_x = blockIdx.x;
    evt.cta_id_y = blockIdx.y;
    evt.is_load = is_load ? 1 : 0;
    auto* ring = (BpfRingBufMap<SampledMemEvent, kRingCapacity>*)pring;
    ring->output(&evt);
}
