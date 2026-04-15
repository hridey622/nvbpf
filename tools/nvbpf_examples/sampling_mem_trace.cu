/*
 * NV-BPF Example: Sampling Memory Trace
 *
 * Samples memory events under user-controlled conditions.
 * Supported knobs:
 *   NVBPF_SAMPLE_EVERY  - keep 1/N qualifying events (default 64)
 *   NVBPF_ADDR_MIN      - inclusive lower address bound (default 0)
 *   NVBPF_ADDR_MAX      - inclusive upper address bound (default UINT64_MAX)
 *   NVBPF_TRACE_LOADS   - 1/0
 *   NVBPF_TRACE_STORES  - 1/0
 *   NVBPF_VERBOSE       - dump sampled events
 *   NVBPF_KERNEL_FILTER - substring filter on kernel name
 */

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_set>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

struct SampledMemEvent {
    uint64_t addr;
    uint32_t sm_id;
    uint32_t warp_id;
    uint16_t cta_id_x;
    uint16_t cta_id_y;
    uint8_t is_load;
    uint8_t _pad[3];
};

BPF_ARRAY(total_loads, uint64_t, 1);
BPF_ARRAY(total_stores, uint64_t, 1);
BPF_ARRAY(sampled_events_kept, uint64_t, 1);
BPF_RINGBUF(sampled_events, SampledMemEvent, 8192);

extern "C" __device__ __noinline__ void smt_trace_mem(int pred,
                                                      uint64_t addr,
                                                      uint64_t pring,
                                                      uint64_t ptotal,
                                                      uint64_t pkept,
                                                      uint32_t sample_every,
                                                      uint64_t addr_min,
                                                      uint64_t addr_max,
                                                      int is_load);

static pthread_mutex_t launch_mutex;
static std::unordered_set<CUfunction> already_instrumented;
static bool verbose = false;
static bool trace_loads = true;
static bool trace_stores = true;
static uint32_t sample_every = 64;
static uint64_t addr_min = 0;
static uint64_t addr_max = ~0ULL;
static std::string kernel_name_filter;

static uint64_t parse_u64_env(const char* name, uint64_t def) {
    const char* env = getenv(name);
    if (!env || !*env) return def;
    return strtoull(env, nullptr, 0);
}

static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);

    for (auto f : related) {
        if (!already_instrumented.insert(f).second) continue;
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        for (auto* instr : instrs) {
            if (trace_loads && instr->isLoad() &&
                instr->getMemorySpace() != InstrType::MemorySpace::CONSTANT) {
                nvbit_insert_call(instr, "smt_trace_mem", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_mref_addr64(instr, 0);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&sampled_events);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&total_loads.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&sampled_events_kept.data[0]);
                nvbit_add_call_arg_const_val32(instr, sample_every);
                nvbit_add_call_arg_const_val64(instr, addr_min);
                nvbit_add_call_arg_const_val64(instr, addr_max);
                nvbit_add_call_arg_const_val32(instr, 1);
            }
            if (trace_stores && instr->isStore()) {
                nvbit_insert_call(instr, "smt_trace_mem", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_mref_addr64(instr, 0);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&sampled_events);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&total_stores.data[0]);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&sampled_events_kept.data[0]);
                nvbit_add_call_arg_const_val32(instr, sample_every);
                nvbit_add_call_arg_const_val64(instr, addr_min);
                nvbit_add_call_arg_const_val64(instr, addr_max);
                nvbit_add_call_arg_const_val32(instr, 0);
            }
        }
    }
}

static void reset_state() {
    total_loads.reset();
    total_stores.reset();
    sampled_events_kept.reset();
    sampled_events.reset();
}

void nvbit_at_init() {
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    pthread_mutex_init(&launch_mutex, nullptr);
    verbose = getenv("NVBPF_VERBOSE") != nullptr;
    trace_loads = parse_u64_env("NVBPF_TRACE_LOADS", 1) != 0;
    trace_stores = parse_u64_env("NVBPF_TRACE_STORES", 1) != 0;
    sample_every = (uint32_t)parse_u64_env("NVBPF_SAMPLE_EVERY", 64);
    if (sample_every == 0) sample_every = 1;
    addr_min = parse_u64_env("NVBPF_ADDR_MIN", 0);
    addr_max = parse_u64_env("NVBPF_ADDR_MAX", ~0ULL);
    if (const char* env = getenv("NVBPF_KERNEL_FILTER")) kernel_name_filter = env;
    printf("[NVBPF SAMPLING_MEM_TRACE] Tool loaded\n");
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (!nvbpf_is_launch_event(cbid)) return;
    CUfunction func = nvbpf_get_launch_func(cbid, params);
    const char* func_name = nvbit_get_func_name(ctx, func);

    if (!is_exit) {
        bool match = kernel_name_filter.empty() ||
                     strstr(func_name, kernel_name_filter.c_str()) != nullptr;
        pthread_mutex_lock(&launch_mutex);
        if (match) {
            instrument_function_if_needed(ctx, func);
            reset_state();
            nvbit_enable_instrumented(ctx, func, true);
        } else {
            nvbit_enable_instrumented(ctx, func, false);
        }
        if (!match) pthread_mutex_unlock(&launch_mutex);
    } else {
        cudaDeviceSynchronize();
        if (!kernel_name_filter.empty() &&
            strstr(func_name, kernel_name_filter.c_str()) == nullptr) {
            return;
        }
        uint64_t loads = *total_loads.lookup(0);
        uint64_t stores = *total_stores.lookup(0);
        uint64_t kept = *sampled_events_kept.lookup(0);
        printf("[NVBPF] %s\n", func_name);
        printf("        loads=%lu stores=%lu sampled=%lu dropped=%lu sample_every=%u\n",
               loads, stores, kept, sampled_events.dropped, sample_every);
        printf("        addr_window=[0x%lx, 0x%lx]\n", addr_min, addr_max);
        if (verbose) {
            uint64_t count = sampled_events.consume([](SampledMemEvent* evt) {
                printf("          %s addr=0x%016lx SM=%u warp=%u CTA=(%u,%u)\n",
                       evt->is_load ? "LOAD " : "STORE",
                       evt->addr, evt->sm_id, evt->warp_id,
                       evt->cta_id_x, evt->cta_id_y);
            });
            printf("        consumed=%lu\n", count);
        } else {
            sampled_events.consume([](SampledMemEvent*) {});
        }
        pthread_mutex_unlock(&launch_mutex);
    }
}

void nvbit_at_term() {
    printf("[NVBPF SAMPLING_MEM_TRACE] Tool terminated\n");
}
