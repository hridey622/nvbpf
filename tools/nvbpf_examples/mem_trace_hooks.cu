/*
 * NV-BPF Example: Memory Tracer - Device Hooks
 * 
 * Device-side instrumentation functions for memory tracing.
 * Compiled with --keep-device-functions and -Xptxas -astoolspatch.
 */

#include <stdint.h>
#include "nvbpf_helpers.h"
#include "nvbpf_maps.h"
#include "nvbpf_types.h"

/* MemAccessEvent struct — must match mem_trace.cu */
struct MemAccessEvent {
    uint64_t addr;
    uint32_t sm_id;
    uint32_t warp_id;
    uint16_t cta_id_x;
    uint16_t cta_id_y;
    uint8_t is_load;
    uint8_t _pad[3];
};

static constexpr int kMemEventCapacity = 16384;

extern "C" __device__ __noinline__ void trace_load(int pred,
                                                   uint64_t addr,
                                                   uint64_t pringbuf,
                                                   uint64_t pcounter) {
    if (!pred) return;
    if (!bpf_is_warp_leader()) return;

    atomicAdd((unsigned long long*)pcounter, 1ULL);

    MemAccessEvent evt;
    evt.addr = addr;
    evt.sm_id = bpf_get_current_sm_id();
    evt.warp_id = bpf_get_current_warp_id();
    evt.cta_id_x = blockIdx.x;
    evt.cta_id_y = blockIdx.y;
    evt.is_load = 1;

    auto* ring = (BpfRingBufMap<MemAccessEvent, kMemEventCapacity>*)pringbuf;
    ring->output(&evt);
}

extern "C" __device__ __noinline__ void trace_store(int pred,
                                                    uint64_t addr,
                                                    uint64_t pringbuf,
                                                    uint64_t pcounter) {
    if (!pred) return;
    if (!bpf_is_warp_leader()) return;

    atomicAdd((unsigned long long*)pcounter, 1ULL);

    MemAccessEvent evt;
    evt.addr = addr;
    evt.sm_id = bpf_get_current_sm_id();
    evt.warp_id = bpf_get_current_warp_id();
    evt.cta_id_x = blockIdx.x;
    evt.cta_id_y = blockIdx.y;
    evt.is_load = 0;

    auto* ring = (BpfRingBufMap<MemAccessEvent, kMemEventCapacity>*)pringbuf;
    ring->output(&evt);
}
