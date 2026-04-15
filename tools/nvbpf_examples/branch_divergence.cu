/*
 * NV-BPF Example: Branch Divergence
 *
 * Counts branch instructions and flags branches executing with fewer than
 * N active lanes per warp.
 */

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_set>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

BPF_ARRAY(total_branches, uint64_t, 1);
BPF_ARRAY(divergent_branches, uint64_t, 1);
BPF_ARRAY(pred_off_branches, uint64_t, 1);
BPF_ARRAY(active_lane_sum, uint64_t, 1);
BPF_ARRAY(active_lane_hist, uint64_t, 33);
BPF_PERCPU_ARRAY(divergent_per_sm, uint64_t, 1);

extern "C" __device__ __noinline__ void bd_trace_branch(int pred,
                                                        uint64_t ptotal,
                                                        uint64_t pdivergent,
                                                        uint64_t ppred_off,
                                                        uint64_t pactive_sum,
                                                        uint64_t phist,
                                                        uint64_t pdivergent_sm,
                                                        uint32_t threshold);

static pthread_mutex_t launch_mutex;
static std::unordered_set<CUfunction> already_instrumented;
static uint32_t active_lane_threshold = 16;
static std::string kernel_name_filter;

static bool is_branch_opcode(const char* opcode) {
    return strncmp(opcode, "BRA", 3) == 0 ||
           strncmp(opcode, "JMP", 3) == 0 ||
           strncmp(opcode, "JMX", 3) == 0 ||
           strncmp(opcode, "BRX", 3) == 0 ||
           strncmp(opcode, "CALL", 4) == 0 ||
           strncmp(opcode, "RET", 3) == 0 ||
           strncmp(opcode, "EXIT", 4) == 0;
}

static void reset_state() {
    total_branches.reset();
    divergent_branches.reset();
    pred_off_branches.reset();
    active_lane_sum.reset();
    active_lane_hist.reset();
    divergent_per_sm.reset();
}

static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);
    for (auto f : related) {
        if (!already_instrumented.insert(f).second) continue;
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        for (auto* instr : instrs) {
            if (!is_branch_opcode(instr->getOpcodeShort())) continue;
            nvbit_insert_call(instr, "bd_trace_branch", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&total_branches.data[0]);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&divergent_branches.data[0]);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&pred_off_branches.data[0]);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&active_lane_sum.data[0]);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&active_lane_hist.data[0]);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&divergent_per_sm.data[0][0]);
            nvbit_add_call_arg_const_val32(instr, active_lane_threshold);
        }
    }
}

void nvbit_at_init() {
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    pthread_mutex_init(&launch_mutex, nullptr);
    if (const char* env = getenv("NVBPF_MIN_ACTIVE_LANES")) {
        active_lane_threshold = (uint32_t)strtoul(env, nullptr, 0);
    }
    if (const char* env = getenv("NVBPF_KERNEL_FILTER")) kernel_name_filter = env;
    printf("[NVBPF BRANCH_DIVERGENCE] Tool loaded\n");
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
        uint64_t total = *total_branches.lookup(0);
        uint64_t divergent = *divergent_branches.lookup(0);
        uint64_t pred_off = *pred_off_branches.lookup(0);
        uint64_t active_sum = *active_lane_sum.lookup(0);
        double avg_active = (total > pred_off) ?
            (double)active_sum / (double)(total - pred_off) : 0.0;
        printf("[NVBPF] %s\n", func_name);
        printf("        total_branches=%lu divergent=%lu predicated_off=%lu threshold=%u avg_active_lanes=%.2f\n",
               total, divergent, pred_off, active_lane_threshold, avg_active);
        printf("        active_lane_hist:");
        for (int i = 0; i <= 32; i++) {
            uint64_t* val = active_lane_hist.lookup(i);
            if (val && *val > 0) printf(" %d:%lu", i, *val);
        }
        printf("\n");
        pthread_mutex_unlock(&launch_mutex);
    }
}

void nvbit_at_term() {
    printf("[NVBPF BRANCH_DIVERGENCE] Tool terminated\n");
}
