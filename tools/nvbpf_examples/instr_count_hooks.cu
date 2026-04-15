/*
 * NV-BPF Example: Instruction Counter - Device Hooks
 *
 * Device-only instrumentation helpers compiled with
 * Compiled with --keep-device-functions and -Xptxas -astoolspatch.
 */

#include <stdint.h>
#include "nvbpf_helpers.h"
#include "nvbpf_types.h"
#include "utils/utils.h"

static constexpr uint32_t kNvbpfMaxSms = 256;

extern "C" __device__ __noinline__ void count_instrs(int predicate,
                                                     uint64_t pcounter,
                                                     uint64_t psm_counters) {
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), predicate);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    const int num_threads = __popc(predicate_mask);

    if (first_laneid != laneid || num_threads == 0) {
        return;
    }

    atomicAdd((unsigned long long*)pcounter, 1ULL);

    uint32_t sm = bpf_get_current_sm_id();
    if (sm < kNvbpfMaxSms) {
        uint64_t* sm_counters = (uint64_t*)psm_counters;
        atomicAdd((unsigned long long*)&sm_counters[sm], 1ULL);
    }
}
