/*
 * NV-BPF Example: Attention Debugger
 *
 * A practical first-pass profiler for transformer-style kernels.
 * It reports instruction mix, memory traffic, and SM activity so you
 * can quickly tell whether a kernel looks compute-bound, memory-bound,
 * or poorly distributed across the GPU.
 *
 * Usage:
 *   make attention_debug.so
 *   LD_PRELOAD=./attention_debug.so ./your_cuda_app
 *
 * Optional:
 *   NVBPF_VERBOSE=1 ...   # dump sampled memory events
 */

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <unordered_set>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

struct AttentionMemEvent {
    uint64_t addr;
    uint32_t sm_id;
    uint32_t warp_id;
    uint16_t cta_id_x;
    uint16_t cta_id_y;
    uint8_t is_load;
    uint8_t _pad[3];
};

BPF_ARRAY(total_instrs, uint64_t, 1);
BPF_ARRAY(tensor_instrs, uint64_t, 1);
BPF_ARRAY(ffma_instrs, uint64_t, 1);
BPF_ARRAY(ldmatrix_instrs, uint64_t, 1);
BPF_ARRAY(cp_async_instrs, uint64_t, 1);
BPF_ARRAY(branch_instrs, uint64_t, 1);
BPF_ARRAY(barrier_instrs, uint64_t, 1);
BPF_ARRAY(load_instrs, uint64_t, 1);
BPF_ARRAY(store_instrs, uint64_t, 1);
BPF_PERCPU_ARRAY(sm_instrs, uint64_t, 1);
BPF_ARRAY(active_sm_bitmap, uint64_t, 4);
BPF_RINGBUF(mem_events, AttentionMemEvent, 4096);

extern "C" __device__ __noinline__ void adbg_count_instr(int pred,
                                                         uint64_t ptotal,
                                                         uint64_t psm_instrs,
                                                         uint64_t pbitmap);
extern "C" __device__ __noinline__ void adbg_count_counter(int pred,
                                                           uint64_t pcounter);
extern "C" __device__ __noinline__ void adbg_trace_mem(int pred,
                                                       uint64_t addr,
                                                       uint64_t pringbuf,
                                                       uint64_t pcounter,
                                                       int is_load);

static pthread_mutex_t launch_mutex;
static std::unordered_set<CUfunction> already_instrumented;
static uint64_t kernel_count = 0;
static bool verbose = false;
static bool current_launch_instrumented = false;
static std::string kernel_name_filter;

static bool opcode_starts_with(const char* opcode, const char* prefix) {
    return strncmp(opcode, prefix, strlen(prefix)) == 0;
}

static bool is_tensor_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "HMMA") ||
           opcode_starts_with(opcode, "MMA") ||
           opcode_starts_with(opcode, "WGMMA") ||
           opcode_starts_with(opcode, "BMMA");
}

static bool is_cp_async_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "CPASYNC") ||
           opcode_starts_with(opcode, "CP") ||
           opcode_starts_with(opcode, "LDGSTS");
}

static bool is_branch_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "BRA") ||
           opcode_starts_with(opcode, "JMP") ||
           opcode_starts_with(opcode, "JMX") ||
           opcode_starts_with(opcode, "BRX") ||
           opcode_starts_with(opcode, "CALL") ||
           opcode_starts_with(opcode, "RET") ||
           opcode_starts_with(opcode, "EXIT");
}

static bool is_barrier_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "BAR") ||
           opcode_starts_with(opcode, "DEPBAR") ||
           opcode_starts_with(opcode, "MEMBAR") ||
           opcode_starts_with(opcode, "ERRBAR");
}

static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);

    for (auto f : related) {
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        for (auto* instr : instrs) {
            const char* opcode = instr->getOpcodeShort();

            nvbit_insert_call(instr, "adbg_count_instr", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&total_instrs.data[0]);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&sm_instrs.data[0][0]);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&active_sm_bitmap.data[0]);

            if (is_tensor_opcode(opcode)) {
                nvbit_insert_call(instr, "adbg_count_counter", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr,
                                               (uint64_t)&tensor_instrs.data[0]);
            }

            if (opcode_starts_with(opcode, "FFMA")) {
                nvbit_insert_call(instr, "adbg_count_counter", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr,
                                               (uint64_t)&ffma_instrs.data[0]);
            }

            if (opcode_starts_with(opcode, "LDMATRIX")) {
                nvbit_insert_call(instr, "adbg_count_counter", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr,
                                               (uint64_t)&ldmatrix_instrs.data[0]);
            }

            if (is_cp_async_opcode(opcode)) {
                nvbit_insert_call(instr, "adbg_count_counter", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr,
                                               (uint64_t)&cp_async_instrs.data[0]);
            }

            if (is_branch_opcode(opcode)) {
                nvbit_insert_call(instr, "adbg_count_counter", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr,
                                               (uint64_t)&branch_instrs.data[0]);
            }

            if (is_barrier_opcode(opcode)) {
                nvbit_insert_call(instr, "adbg_count_counter", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr,
                                               (uint64_t)&barrier_instrs.data[0]);
            }

            if (instr->isLoad() &&
                instr->getMemorySpace() != InstrType::MemorySpace::CONSTANT) {
                nvbit_insert_call(instr, "adbg_trace_mem", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_mref_addr64(instr, 0);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&mem_events);
                nvbit_add_call_arg_const_val64(instr,
                                               (uint64_t)&load_instrs.data[0]);
                nvbit_add_call_arg_const_val32(instr, 1);
            }

            if (instr->isStore()) {
                nvbit_insert_call(instr, "adbg_trace_mem", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_mref_addr64(instr, 0);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&mem_events);
                nvbit_add_call_arg_const_val64(instr,
                                               (uint64_t)&store_instrs.data[0]);
                nvbit_add_call_arg_const_val32(instr, 0);
            }
        }
    }
}

static void reset_state() {
    total_instrs.reset();
    tensor_instrs.reset();
    ffma_instrs.reset();
    ldmatrix_instrs.reset();
    cp_async_instrs.reset();
    branch_instrs.reset();
    barrier_instrs.reset();
    load_instrs.reset();
    store_instrs.reset();
    sm_instrs.reset();
    active_sm_bitmap.reset();
    mem_events.reset();
}

static void print_heuristics(uint64_t total,
                             uint64_t tensor,
                             uint64_t ffma,
                             uint64_t loads,
                             uint64_t stores,
                             uint64_t branches,
                             uint64_t barriers,
                             int active_sms) {
    printf("        Heuristics:\n");
    if (active_sms <= 1) {
        printf("          Low SM utilization for this launch.\n");
    }
    if (total > 0 && (loads + stores) * 3 > total) {
        printf("          Memory traffic is large relative to executed instructions.\n");
    }
    if (total > 0 && (branches + barriers) * 10 > total) {
        printf("          Control-flow or synchronization overhead looks non-trivial.\n");
    }
    if (tensor == 0 && ffma == 0) {
        printf("          Little obvious math throughput in the instruction mix.\n");
    } else if (tensor > 0 && loads + stores > tensor * 2) {
        printf("          Tensor/math ops exist, but memory traffic may be limiting them.\n");
    }
}

void nvbit_at_init() {
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    pthread_mutex_init(&launch_mutex, nullptr);
    verbose = getenv("NVBPF_VERBOSE") != nullptr;
    if (const char* env = getenv("NVBPF_KERNEL_FILTER")) {
        kernel_name_filter = env;
    }
    printf("[NVBPF ATTENTION_DEBUG] Tool loaded\n");
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (!nvbpf_is_launch_event(cbid)) return;

    CUfunction func = nvbpf_get_launch_func(cbid, params);
    const char* func_name = nvbit_get_func_name(ctx, func);

    if (!is_exit) {
        bool should_instrument =
            kernel_name_filter.empty() ||
            strstr(func_name, kernel_name_filter.c_str()) != nullptr;

        pthread_mutex_lock(&launch_mutex);
        current_launch_instrumented = should_instrument;
        if (should_instrument) {
            instrument_function_if_needed(ctx, func);
            reset_state();
            nvbit_enable_instrumented(ctx, func, true);
        } else {
            nvbit_enable_instrumented(ctx, func, false);
        }
    } else {
        cudaDeviceSynchronize();

        if (!current_launch_instrumented) {
            pthread_mutex_unlock(&launch_mutex);
            return;
        }

        uint64_t total = *total_instrs.lookup(0);
        uint64_t tensor = *tensor_instrs.lookup(0);
        uint64_t ffma = *ffma_instrs.lookup(0);
        uint64_t ldmatrix = *ldmatrix_instrs.lookup(0);
        uint64_t cp_async = *cp_async_instrs.lookup(0);
        uint64_t branches = *branch_instrs.lookup(0);
        uint64_t barriers = *barrier_instrs.lookup(0);
        uint64_t loads = *load_instrs.lookup(0);
        uint64_t stores = *store_instrs.lookup(0);

        int active_sms = 0;
        for (int word = 0; word < 4; word++) {
            uint64_t* bm = active_sm_bitmap.lookup(word);
            if (bm) active_sms += __builtin_popcountll(*bm);
        }

        printf("[NVBPF] Kernel %lu: %s\n", kernel_count++,
               func_name);
        printf("        Total warp instructions: %lu\n", total);
        printf("        Math mix: tensor=%lu ffma=%lu ldmatrix=%lu cp_async=%lu\n",
               tensor, ffma, ldmatrix, cp_async);
        printf("        Control: branches=%lu barriers=%lu\n",
               branches, barriers);
        printf("        Memory: loads=%lu stores=%lu\n", loads, stores);
        printf("        Active SMs: %d\n", active_sms);

        printf("        Per-SM totals:\n");
        for (int sm = 0; sm < 128; sm++) {
            uint64_t* val = sm_instrs.lookup_sm(sm, 0);
            if (val && *val > 0) {
                printf("          SM %3d: %8lu instrs\n", sm, *val);
            }
        }

        print_heuristics(total, tensor, ffma, loads, stores, branches,
                         barriers, active_sms);

        if (verbose) {
            uint64_t events = mem_events.consume([](AttentionMemEvent* evt) {
                printf("          %s addr=0x%016lx SM=%u warp=%u CTA=(%u,%u)\n",
                       evt->is_load ? "LOAD " : "STORE",
                       evt->addr, evt->sm_id, evt->warp_id,
                       evt->cta_id_x, evt->cta_id_y);
            });
            printf("        Memory samples: %lu\n", events);
        } else {
            mem_events.consume([](AttentionMemEvent*) {});
        }

        if (mem_events.dropped > 0) {
            printf("        [WARNING] %lu memory samples dropped\n",
                   mem_events.dropped);
        }

        pthread_mutex_unlock(&launch_mutex);
    }
}

void nvbit_at_term() {
    printf("[NVBPF ATTENTION_DEBUG] Tool terminated\n");
}
