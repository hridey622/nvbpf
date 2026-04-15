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

#include <pthread.h>
#include <unordered_set>

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
 * 2. Instrumentation
 * ============================================ */

extern "C" __device__ __noinline__ void trace_load(int pred,
                                                   uint64_t addr,
                                                   uint64_t pringbuf,
                                                   uint64_t pcounter);
extern "C" __device__ __noinline__ void trace_store(int pred,
                                                    uint64_t addr,
                                                    uint64_t pringbuf,
                                                    uint64_t pcounter);

static pthread_mutex_t launch_mutex;
static std::unordered_set<CUfunction> already_instrumented;

static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);

    for (auto f : related) {
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        for (auto* instr : instrs) {
            if (instr->isLoad() &&
                instr->getMemorySpace() != InstrType::MemorySpace::CONSTANT) {
                nvbit_insert_call(instr, "trace_load", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_mref_addr64(instr, 0);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&mem_events);
                nvbit_add_call_arg_const_val64(instr,
                                               (uint64_t)&load_count.data[0]);
            }

            if (instr->isStore()) {
                nvbit_insert_call(instr, "trace_store", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_mref_addr64(instr, 0);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&mem_events);
                nvbit_add_call_arg_const_val64(instr,
                                               (uint64_t)&store_count.data[0]);
            }
        }
    }
}

/* ============================================
 * 3. NVBit Callbacks
 * ============================================ */

static uint64_t kernel_count = 0;
static bool verbose = false;

void nvbit_at_init() {
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    pthread_mutex_init(&launch_mutex, nullptr);
    
    // Check for verbose mode
    if (getenv("NVBPF_VERBOSE")) {
        verbose = true;
    }
    
    printf("[NVBPF MEM_TRACE] Tool loaded\n");
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (!nvbpf_is_launch_event(cbid)) return;
    
    CUfunction func = nvbpf_get_launch_func(cbid, params);
    
    if (!is_exit) {
        pthread_mutex_lock(&launch_mutex);
        instrument_function_if_needed(ctx, func);
        load_count.reset();
        store_count.reset();
        mem_events.reset();
        nvbit_enable_instrumented(ctx, func, true);
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
        pthread_mutex_unlock(&launch_mutex);
    }
}

void nvbit_at_term() {
    printf("[NVBPF MEM_TRACE] Tool terminated\n");
}
