/*
 * NV-BPF Example: Tail Fragment Tracker
 *
 * Estimates tail/edge inefficiency by counting instrumented warp sites that
 * execute with fewer active lanes than the current warp width. This is an
 * approximation of ragged-tile / edge-tile waste rather than a hardware stall
 * counter.
 */

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unordered_set>
#include <vector>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

static constexpr int SK_LOAD = 0;
static constexpr int SK_STORE = 1;
static constexpr int SK_MATH = 2;
static constexpr int SK_BRANCH = 3;
static constexpr int SK_ALL = 4;
static constexpr int SK_COUNT = 5;

BPF_ARRAY(total_sites, uint64_t, SK_COUNT);
BPF_ARRAY(partial_sites, uint64_t, SK_COUNT);
BPF_ARRAY(dead_sites, uint64_t, SK_COUNT);
BPF_ARRAY(low_lane_sites, uint64_t, SK_COUNT);
BPF_ARRAY(wasted_lanes, uint64_t, SK_COUNT);
BPF_ARRAY(active_lane_sum, uint64_t, SK_COUNT);
BPF_ARRAY(warp_lane_sum, uint64_t, SK_COUNT);
BPF_ARRAY(active_lane_hist, uint64_t, 33);

extern "C" __device__ __noinline__ void tf_trace_site(int pred,
                                                      uint64_t ptotal,
                                                      uint64_t ppartial,
                                                      uint64_t pdead,
                                                      uint64_t plow,
                                                      uint64_t pwaste,
                                                      uint64_t pactive_sum,
                                                      uint64_t pwarp_sum,
                                                      uint64_t phist,
                                                      uint32_t threshold,
                                                      uint32_t kind);

static pthread_mutex_t launch_mutex;
static std::unordered_set<CUfunction> already_instrumented;
static std::string kernel_name_filter;
static uint32_t active_lane_threshold = 16;
static bool verbose = false;
static bool full_names = false;
static uint64_t matched_launches = 0;

struct TailSummary {
    std::string kernel_name;
    uint64_t launches = 0;
    uint64_t total[SK_COUNT] = {};
    uint64_t partial[SK_COUNT] = {};
    uint64_t dead[SK_COUNT] = {};
    uint64_t low[SK_COUNT] = {};
    uint64_t waste[SK_COUNT] = {};
    uint64_t active_sum[SK_COUNT] = {};
    uint64_t warp_sum[SK_COUNT] = {};
};

static std::vector<TailSummary> summaries;

static bool opcode_starts_with(const char* opcode, const char* prefix) {
    return strncmp(opcode, prefix, strlen(prefix)) == 0;
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

static bool is_math_opcode(const char* opcode) {
    return opcode_starts_with(opcode, "FFMA") ||
           opcode_starts_with(opcode, "FADD") ||
           opcode_starts_with(opcode, "FMUL") ||
           opcode_starts_with(opcode, "HFMA") ||
           opcode_starts_with(opcode, "HMMA") ||
           opcode_starts_with(opcode, "MMA") ||
           opcode_starts_with(opcode, "WGMMA") ||
           opcode_starts_with(opcode, "IMMA") ||
           opcode_starts_with(opcode, "BMMA") ||
           opcode_starts_with(opcode, "DMMA") ||
           opcode_starts_with(opcode, "IMAD") ||
           opcode_starts_with(opcode, "IADD3");
}

static int site_kind(Instr* instr) {
    if (instr->isLoad() &&
        instr->getMemorySpace() != InstrType::MemorySpace::CONSTANT) {
        return SK_LOAD;
    }
    if (instr->isStore()) {
        return SK_STORE;
    }
    const char* opcode = instr->getOpcodeShort();
    if (is_branch_opcode(opcode)) {
        return SK_BRANCH;
    }
    if (is_math_opcode(opcode)) {
        return SK_MATH;
    }
    return -1;
}

static const char* kind_name(int kind) {
    switch (kind) {
        case SK_LOAD: return "load";
        case SK_STORE: return "store";
        case SK_MATH: return "math";
        case SK_BRANCH: return "branch";
        case SK_ALL: return "all";
        default: return "unknown";
    }
}

static std::string compact_kernel_name(const std::string& raw) {
    if (full_names) return raw;
    std::string name = raw;
    if (name.rfind("void ", 0) == 0) {
        name = name.substr(5);
    }
    size_t paren = name.find('(');
    if (paren != std::string::npos) {
        name = name.substr(0, paren);
    }
    if (name.size() <= 56) return name;
    return name.substr(0, 24) + "..." + name.substr(name.size() - 24);
}

static double pct(uint64_t num, uint64_t den) {
    return den == 0 ? 0.0 : 100.0 * (double)num / (double)den;
}

static double avg_active(uint64_t active_sum_value, uint64_t total_sites_value) {
    return total_sites_value == 0 ? 0.0
                                  : (double)active_sum_value / (double)total_sites_value;
}

static TailSummary* find_summary(const char* func_name) {
    for (auto& summary : summaries) {
        if (summary.kernel_name == func_name) {
            return &summary;
        }
    }
    return nullptr;
}

static void reset_state() {
    total_sites.reset();
    partial_sites.reset();
    dead_sites.reset();
    low_lane_sites.reset();
    wasted_lanes.reset();
    active_lane_sum.reset();
    warp_lane_sum.reset();
    active_lane_hist.reset();
}

static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);
    for (auto f : related) {
        if (!already_instrumented.insert(f).second) continue;
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        for (auto* instr : instrs) {
            int kind = site_kind(instr);
            if (kind < 0) continue;
            nvbit_insert_call(instr, "tf_trace_site", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&total_sites.data[0]);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&partial_sites.data[0]);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&dead_sites.data[0]);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&low_lane_sites.data[0]);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&wasted_lanes.data[0]);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&active_lane_sum.data[0]);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&warp_lane_sum.data[0]);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&active_lane_hist.data[0]);
            nvbit_add_call_arg_const_val32(instr, active_lane_threshold);
            nvbit_add_call_arg_const_val32(instr, (uint32_t)kind);
        }
    }
}

void nvbit_at_init() {
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    pthread_mutex_init(&launch_mutex, nullptr);
    if (const char* env = getenv("NVBPF_KERNEL_FILTER")) kernel_name_filter = env;
    if (const char* env = getenv("NVBPF_TAIL_ACTIVE_LANES")) {
        active_lane_threshold = (uint32_t)strtoul(env, nullptr, 0);
    }
    verbose = getenv("NVBPF_VERBOSE") != nullptr;
    full_names = getenv("NVBPF_FULL_NAMES") != nullptr;
    printf("[NVBPF TAIL_FRAGMENT_TRACKER] Tool loaded\n");
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
        matched_launches++;

        uint64_t launch_total[SK_COUNT] = {};
        uint64_t launch_partial[SK_COUNT] = {};
        uint64_t launch_dead[SK_COUNT] = {};
        uint64_t launch_low[SK_COUNT] = {};
        uint64_t launch_waste[SK_COUNT] = {};
        uint64_t launch_active_sum[SK_COUNT] = {};
        uint64_t launch_warp_sum[SK_COUNT] = {};
        for (int i = 0; i < SK_COUNT; i++) {
            uint64_t* total = total_sites.lookup(i);
            uint64_t* partial = partial_sites.lookup(i);
            uint64_t* dead = dead_sites.lookup(i);
            uint64_t* low = low_lane_sites.lookup(i);
            uint64_t* waste = wasted_lanes.lookup(i);
            uint64_t* active = active_lane_sum.lookup(i);
            uint64_t* warp = warp_lane_sum.lookup(i);
            launch_total[i] = total ? *total : 0;
            launch_partial[i] = partial ? *partial : 0;
            launch_dead[i] = dead ? *dead : 0;
            launch_low[i] = low ? *low : 0;
            launch_waste[i] = waste ? *waste : 0;
            launch_active_sum[i] = active ? *active : 0;
            launch_warp_sum[i] = warp ? *warp : 0;
        }

        if (verbose) {
            printf("[NVBPF] tail_fragment kernel=%s\n", func_name);
            printf("        all: sites=%lu partial=%.2f%% low<%u=%.2f%% dead=%.2f%% waste=%.2f%% avg_active=%.2f\n",
                   launch_total[SK_ALL],
                   pct(launch_partial[SK_ALL], launch_total[SK_ALL]),
                   active_lane_threshold,
                   pct(launch_low[SK_ALL], launch_total[SK_ALL]),
                   pct(launch_dead[SK_ALL], launch_total[SK_ALL]),
                   pct(launch_waste[SK_ALL], launch_warp_sum[SK_ALL]),
                   avg_active(launch_active_sum[SK_ALL], launch_total[SK_ALL]));
            for (int kind = 0; kind < SK_ALL; kind++) {
                if (launch_total[kind] == 0) continue;
                printf("        %-6s sites=%lu partial=%.2f%% low<%u=%.2f%% dead=%.2f%% waste=%.2f%% avg_active=%.2f\n",
                       kind_name(kind), launch_total[kind],
                       pct(launch_partial[kind], launch_total[kind]),
                       active_lane_threshold,
                       pct(launch_low[kind], launch_total[kind]),
                       pct(launch_dead[kind], launch_total[kind]),
                       pct(launch_waste[kind], launch_warp_sum[kind]),
                       avg_active(launch_active_sum[kind], launch_total[kind]));
            }
            printf("        active_lane_hist:");
            for (int lanes = 0; lanes <= 32; lanes++) {
                uint64_t* val = active_lane_hist.lookup(lanes);
                if (val && *val > 0) printf(" %d:%lu", lanes, *val);
            }
            printf("\n");
        } else {
            TailSummary* summary = find_summary(func_name);
            if (summary == nullptr) {
                TailSummary fresh{};
                fresh.kernel_name = func_name;
                summaries.push_back(fresh);
                summary = &summaries.back();
            }
            summary->launches++;
            for (int i = 0; i < SK_COUNT; i++) {
                summary->total[i] += launch_total[i];
                summary->partial[i] += launch_partial[i];
                summary->dead[i] += launch_dead[i];
                summary->low[i] += launch_low[i];
                summary->waste[i] += launch_waste[i];
                summary->active_sum[i] += launch_active_sum[i];
                summary->warp_sum[i] += launch_warp_sum[i];
            }
        }
        pthread_mutex_unlock(&launch_mutex);
    }
}

void nvbit_at_term() {
    if (!verbose) {
        printf("[NVBPF TAIL_FRAGMENT_TRACKER] matched_launches=%lu threshold=%u unique_kernels=%zu\n",
               matched_launches, active_lane_threshold, summaries.size());
        for (const auto& summary : summaries) {
            uint64_t mem_total = summary.total[SK_LOAD] + summary.total[SK_STORE];
            uint64_t mem_partial = summary.partial[SK_LOAD] + summary.partial[SK_STORE];
            printf("  x%-3lu %-32s | sites=%-8lu partial=%5.2f%% low=%5.2f%% dead=%5.2f%% waste=%5.2f%% avg=%5.2f | math_p=%5.2f%% mem_p=%5.2f%% br_p=%5.2f%%\n",
                   summary.launches,
                   compact_kernel_name(summary.kernel_name).c_str(),
                   summary.total[SK_ALL],
                   pct(summary.partial[SK_ALL], summary.total[SK_ALL]),
                   pct(summary.low[SK_ALL], summary.total[SK_ALL]),
                   pct(summary.dead[SK_ALL], summary.total[SK_ALL]),
                   pct(summary.waste[SK_ALL], summary.warp_sum[SK_ALL]),
                   avg_active(summary.active_sum[SK_ALL], summary.total[SK_ALL]),
                   pct(summary.partial[SK_MATH], summary.total[SK_MATH]),
                   pct(mem_partial, mem_total),
                   pct(summary.partial[SK_BRANCH], summary.total[SK_BRANCH]));
        }
    }
    printf("[NVBPF TAIL_FRAGMENT_TRACKER] Tool terminated\n");
}
