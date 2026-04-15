/*
 * NV-BPF Example: Attention Trace
 *
 * Stage-oriented summary for attention-like workloads.
 */

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <unordered_set>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

BPF_ARRAY(total_instrs, uint64_t, 1);
BPF_ARRAY(ffma_instrs, uint64_t, 1);
BPF_ARRAY(tensor_instrs, uint64_t, 1);
BPF_ARRAY(load_instrs, uint64_t, 1);
BPF_ARRAY(store_instrs, uint64_t, 1);
BPF_ARRAY(branch_instrs, uint64_t, 1);
BPF_PERCPU_ARRAY(sm_instrs, uint64_t, 1);

extern "C" __device__ __noinline__ void at_count_instr(int pred,
                                                       uint64_t ptotal,
                                                       uint64_t psm_instrs);
extern "C" __device__ __noinline__ void at_count_counter(int pred,
                                                         uint64_t pcounter);

enum Stage {
    STAGE_QK = 0,
    STAGE_SOFTMAX = 1,
    STAGE_PV = 2,
    STAGE_SCALE_MASK = 3,
    STAGE_UNKNOWN = 4,
    STAGE_COUNT = 5,
};

static const char* kStageNames[STAGE_COUNT] = {
    "qk_matmul", "softmax", "pv_matmul", "scale_mask", "unknown"
};

static pthread_mutex_t launch_mutex;
static std::unordered_set<CUfunction> already_instrumented;
static std::string filter_csv = "attention,attn,softmax,sgemm,bmm,gemm,flash";
static uint64_t stage_launches[STAGE_COUNT] = {};
static uint64_t stage_instr_totals[STAGE_COUNT] = {};

static bool opcode_starts_with(const char* opcode, const char* prefix) {
    return strncmp(opcode, prefix, strlen(prefix)) == 0;
}

static bool csv_match(const char* name, const std::string& csv) {
    size_t start = 0;
    while (start < csv.size()) {
        size_t end = csv.find(',', start);
        if (end == std::string::npos) end = csv.size();
        std::string tok = csv.substr(start, end - start);
        if (!tok.empty() && strstr(name, tok.c_str()) != nullptr) return true;
        start = end + 1;
    }
    return false;
}

static Stage classify_stage(const char* name) {
    if (strstr(name, "softmax")) return STAGE_SOFTMAX;
    if (strstr(name, "mul") || strstr(name, "scale") || strstr(name, "mask")) return STAGE_SCALE_MASK;
    if (strstr(name, "sgemm") || strstr(name, "gemm") || strstr(name, "bmm")) {
        if (strstr(name, "_tn")) return STAGE_QK;
        if (strstr(name, "_nn")) return STAGE_PV;
    }
    return STAGE_UNKNOWN;
}

static void reset_state() {
    total_instrs.reset();
    ffma_instrs.reset();
    tensor_instrs.reset();
    load_instrs.reset();
    store_instrs.reset();
    branch_instrs.reset();
    sm_instrs.reset();
}

static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);

    for (auto f : related) {
        if (!already_instrumented.insert(f).second) continue;
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        for (auto* instr : instrs) {
            const char* opcode = instr->getOpcodeShort();
            nvbit_insert_call(instr, "at_count_instr", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&total_instrs.data[0]);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&sm_instrs.data[0][0]);

            if (opcode_starts_with(opcode, "FFMA")) {
                nvbit_insert_call(instr, "at_count_counter", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&ffma_instrs.data[0]);
            }
            if (opcode_starts_with(opcode, "HMMA") ||
                opcode_starts_with(opcode, "MMA") ||
                opcode_starts_with(opcode, "WGMMA")) {
                nvbit_insert_call(instr, "at_count_counter", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tensor_instrs.data[0]);
            }
            if (instr->isLoad() &&
                instr->getMemorySpace() != InstrType::MemorySpace::CONSTANT) {
                nvbit_insert_call(instr, "at_count_counter", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&load_instrs.data[0]);
            }
            if (instr->isStore()) {
                nvbit_insert_call(instr, "at_count_counter", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&store_instrs.data[0]);
            }
            if (opcode_starts_with(opcode, "BRA") ||
                opcode_starts_with(opcode, "JMP") ||
                opcode_starts_with(opcode, "BRX")) {
                nvbit_insert_call(instr, "at_count_counter", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&branch_instrs.data[0]);
            }
        }
    }
}

void nvbit_at_init() {
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    pthread_mutex_init(&launch_mutex, nullptr);
    if (const char* env = getenv("NVBPF_ATTENTION_FILTER")) filter_csv = env;
    printf("[NVBPF ATTENTION_TRACE] Tool loaded\n");
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (!nvbpf_is_launch_event(cbid)) return;
    CUfunction func = nvbpf_get_launch_func(cbid, params);
    const char* func_name = nvbit_get_func_name(ctx, func);

    if (!is_exit) {
        bool match = csv_match(func_name, filter_csv);
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
        if (!csv_match(func_name, filter_csv)) return;

        Stage stage = classify_stage(func_name);
        uint64_t total = *total_instrs.lookup(0);
        stage_launches[stage]++;
        stage_instr_totals[stage] += total;

        uint64_t ffma = *ffma_instrs.lookup(0);
        uint64_t tensor = *tensor_instrs.lookup(0);
        uint64_t loads = *load_instrs.lookup(0);
        uint64_t stores = *store_instrs.lookup(0);
        uint64_t branches = *branch_instrs.lookup(0);
        printf("[NVBPF] stage=%s kernel=%s\n", kStageNames[stage], func_name);
        printf("        instrs=%lu ffma=%lu tensor=%lu loads=%lu stores=%lu branches=%lu\n",
               total, ffma, tensor, loads, stores, branches);
        pthread_mutex_unlock(&launch_mutex);
    }
}

void nvbit_at_term() {
    printf("[NVBPF ATTENTION_TRACE] Stage totals:\n");
    for (int i = 0; i < STAGE_COUNT; i++) {
        if (stage_launches[i] > 0) {
            printf("  %s: launches=%lu instrs=%lu\n",
                   kStageNames[i], stage_launches[i], stage_instr_totals[i]);
        }
    }
    printf("[NVBPF ATTENTION_TRACE] Tool terminated\n");
}
