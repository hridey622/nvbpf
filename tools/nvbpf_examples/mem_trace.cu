/*
 * NV-BPF Example: Memory Tracer
 * 
 * Traces memory load/store operations using the NV-BPF
 * tracepoint interface.
 * 
 * Usage:
 *   make
 *   LD_PRELOAD=./mem_trace.so ./your_cuda_app
 */

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

/* ============================================
 * 1. Define Maps
 * ============================================ */

// Memory access event for ring buffer
struct MemAccessEvent {
    uint64_t addr;
    uint32_t sm_id;
    uint32_t warp_id;
    uint16_t cta_id_x;
    uint16_t cta_id_y;
    uint8_t is_load;
    uint8_t _pad[3];
};

// Ring buffer for streaming events to CPU
BPF_RINGBUF(mem_events, MemAccessEvent, 16384);

// Counters for summary
BPF_ARRAY(load_count, uint64_t, 1);
BPF_ARRAY(store_count, uint64_t, 1);

/* ============================================
 * 2. Define Hooks
 * ============================================ */

// Trace memory loads
SEC_TRACEPOINT_MEM_LOAD(trace_load) {
    BPF_REQUIRE_PRED(pred);
    BPF_WARP_LEADER_ONLY();
    
    // Count loads
    load_count.atomic_inc(0);
    
    // Record event to ring buffer
    MemAccessEvent evt;
    evt.addr = addr;
    evt.sm_id = bpf_get_current_sm_id();
    evt.warp_id = bpf_get_current_warp_id();
    evt.cta_id_x = blockIdx.x;
    evt.cta_id_y = blockIdx.y;
    evt.is_load = 1;
    
    mem_events.output(&evt);
}

// Trace memory stores
SEC_TRACEPOINT_MEM_STORE(trace_store) {
    BPF_REQUIRE_PRED(pred);
    BPF_WARP_LEADER_ONLY();
    
    // Count stores
    store_count.atomic_inc(0);
    
    // Record event
    MemAccessEvent evt;
    evt.addr = addr;
    evt.sm_id = bpf_get_current_sm_id();
    evt.warp_id = bpf_get_current_warp_id();
    evt.cta_id_x = blockIdx.x;
    evt.cta_id_y = blockIdx.y;
    evt.is_load = 0;
    
    mem_events.output(&evt);
}

/* ============================================
 * 3. NVBit Callbacks
 * ============================================ */

static uint64_t kernel_count = 0;
static bool verbose = false;

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    
    // Check for verbose mode
    if (getenv("NVBPF_VERBOSE")) {
        verbose = true;
    }
    
    printf("[NVBPF MEM_TRACE] Tool loaded\n");
    nvbpf::nvbpf_print_hooks();
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (!nvbpf_is_launch_event(cbid)) return;
    
    CUfunction func = nvbpf_get_launch_func(cbid, params);
    
    if (!is_exit) {
        nvbpf_attach_hooks(ctx, func);
    } else {
        cudaDeviceSynchronize();
        
        uint64_t loads = *load_count.lookup(0);
        uint64_t stores = *store_count.lookup(0);
        
        printf("[NVBPF] Kernel %lu: %s\n", kernel_count++, 
               nvbit_get_func_name(ctx, func));
        printf("        Loads: %lu, Stores: %lu\n", loads, stores);
        
        // Process ring buffer events
        if (verbose) {
            uint64_t events_processed = mem_events.consume([](MemAccessEvent* evt) {
                printf("  %s addr=0x%016lx SM=%u warp=%u CTA=(%u,%u)\n",
                       evt->is_load ? "LOAD " : "STORE",
                       evt->addr, evt->sm_id, evt->warp_id,
                       evt->cta_id_x, evt->cta_id_y);
            });
            printf("        Ring buffer events: %lu\n", events_processed);
        } else {
            // Just drain the buffer
            mem_events.consume([](MemAccessEvent*) {});
        }
        
        if (mem_events.dropped > 0) {
            printf("        [WARNING] %lu events dropped (buffer full)\n", 
                   mem_events.dropped);
        }
        
        // Reset for next kernel
        load_count.reset();
        store_count.reset();
        mem_events.reset();
    }
}

void nvbit_at_term() {
    printf("[NVBPF MEM_TRACE] Tool terminated\n");
}

void nvbit_at_ctx_init(CUcontext ctx) {}
void nvbit_at_ctx_term(CUcontext ctx) {}
