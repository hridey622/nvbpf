/*
 * NV-BPF Example: Attention Trace - Device Hooks
 */

#include <stdint.h>
#include "nvbpf_helpers.h"
#include "utils/utils.h"

static constexpr uint32_t kNvbpfMaxSms = 256;

extern "C" __device__ __noinline__ void at_count_instr(int pred,
                                                       uint64_t ptotal,
                                                       uint64_t psm_instrs) {
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), pred);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    const int num_threads = __popc(predicate_mask);
    if (first_laneid != laneid || num_threads == 0) return;

    atomicAdd((unsigned long long*)ptotal, 1ULL);
    uint32_t sm = bpf_get_current_sm_id();
    if (sm >= kNvbpfMaxSms) return;
    uint64_t* sm_instrs = (uint64_t*)psm_instrs;
    atomicAdd((unsigned long long*)&sm_instrs[sm], 1ULL);
}

extern "C" __device__ __noinline__ void at_count_counter(int pred,
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
