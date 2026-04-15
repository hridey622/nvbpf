/*
 * NV-BPF Example: Attention Debugger - Device Hooks
 *
 * Device-side helpers for instruction, memory, and SM activity counting.
 */

#include <stdint.h>
#include "nvbpf_helpers.h"
#include "nvbpf_maps.h"
#include "nvbpf_types.h"
#include "utils/utils.h"

struct AttentionMemEvent {
    uint64_t addr;
    uint32_t sm_id;
    uint32_t warp_id;
    uint16_t cta_id_x;
    uint16_t cta_id_y;
    uint8_t is_load;
    uint8_t _pad[3];
};

static constexpr uint32_t kNvbpfMaxSms = 256;
static constexpr uint32_t kBitmapWords = 4;
static constexpr int kMemEventCapacity = 4096;

extern "C" __device__ __noinline__ void adbg_count_instr(int pred,
                                                         uint64_t ptotal,
                                                         uint64_t psm_instrs,
                                                         uint64_t pbitmap) {
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), pred);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    const int num_threads = __popc(predicate_mask);

    if (first_laneid != laneid || num_threads == 0) {
        return;
    }

    atomicAdd((unsigned long long*)ptotal, 1ULL);

    uint32_t sm = bpf_get_current_sm_id();
    if (sm >= kNvbpfMaxSms) {
        return;
    }

    uint64_t* sm_instrs = (uint64_t*)psm_instrs;
    atomicAdd((unsigned long long*)&sm_instrs[sm], 1ULL);

    uint32_t word = sm / 64;
    uint64_t bit = 1ULL << (sm % 64);
    if (word < kBitmapWords) {
        uint64_t* bitmap = (uint64_t*)pbitmap;
        atomicOr((unsigned long long*)&bitmap[word], bit);
    }
}

extern "C" __device__ __noinline__ void adbg_count_counter(int pred,
                                                           uint64_t pcounter) {
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), pred);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    const int num_threads = __popc(predicate_mask);

    if (first_laneid == laneid && num_threads > 0) {
        atomicAdd((unsigned long long*)pcounter, 1ULL);
    }
}

extern "C" __device__ __noinline__ void adbg_trace_mem(int pred,
                                                       uint64_t addr,
                                                       uint64_t pringbuf,
                                                       uint64_t pcounter,
                                                       int is_load) {
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), pred);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    const int num_threads = __popc(predicate_mask);

    if (first_laneid != laneid || num_threads == 0) {
        return;
    }

    atomicAdd((unsigned long long*)pcounter, 1ULL);

    AttentionMemEvent evt;
    evt.addr = addr;
    evt.sm_id = bpf_get_current_sm_id();
    evt.warp_id = bpf_get_current_warp_id();
    evt.cta_id_x = blockIdx.x;
    evt.cta_id_y = blockIdx.y;
    evt.is_load = is_load ? 1 : 0;

    auto* ring = (BpfRingBufMap<AttentionMemEvent, kMemEventCapacity>*)pringbuf;
    ring->output(&evt);
}
