/*
 * NV-BPF Example: Bank Conflict Suspicion
 *
 * Static shared-memory access heuristic. This is intentionally phrased as a
 * suspicion score rather than a ground-truth bank-conflict counter.
 */

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

struct BankSummary {
    std::string kernel_name;
    uint64_t launches = 0;
    uint64_t shared_loads = 0;
    uint64_t shared_stores = 0;
    uint64_t ldmatrix = 0;
    uint64_t cp_async = 0;
    uint64_t tensor_ops = 0;
    uint64_t vectorish_shared = 0;
    uint64_t global_to_shared = 0;
    int risk_score = 0;
    std::string label;
};

static pthread_mutex_t launch_mutex;
static std::vector<BankSummary> summaries;
static std::string kernel_name_filter;
static bool verbose = false;
static bool full_names = false;
static uint64_t matched_launches = 0;

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
           opcode_starts_with(opcode, "LDGSTS");
}

static bool is_vectorish_sass(const char* sass) {
    return strstr(sass, ".128") != nullptr ||
           strstr(sass, ".64") != nullptr ||
           strstr(sass, "LDMATRIX") != nullptr;
}

static std::string compact_kernel_name(const std::string& raw) {
    if (full_names) return raw;
    std::string name = raw;
    if (name.rfind("void ", 0) == 0) name = name.substr(5);
    size_t paren = name.find('(');
    if (paren != std::string::npos) name = name.substr(0, paren);
    if (name.size() <= 56) return name;
    return name.substr(0, 24) + "..." + name.substr(name.size() - 24);
}

static BankSummary* find_summary(const char* func_name) {
    for (auto& summary : summaries) {
        if (summary.kernel_name == func_name) return &summary;
    }
    return nullptr;
}

static BankSummary analyze_bank_conflicts(CUcontext ctx, CUfunction func,
                                          const char* func_name) {
    BankSummary summary{};
    summary.kernel_name = func_name;

    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);

    for (auto f : related) {
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        for (auto* instr : instrs) {
            const char* opcode = instr->getOpcodeShort();
            if (instr->isLoad() && instr->getMemorySpace() == InstrType::MemorySpace::SHARED) {
                summary.shared_loads++;
                if (is_vectorish_sass(instr->getSass())) summary.vectorish_shared++;
            }
            if (instr->isStore() && instr->getMemorySpace() == InstrType::MemorySpace::SHARED) {
                summary.shared_stores++;
                if (is_vectorish_sass(instr->getSass())) summary.vectorish_shared++;
            }
            if (instr->getMemorySpace() == InstrType::MemorySpace::GLOBAL_TO_SHARED) {
                summary.global_to_shared++;
            }
            if (opcode_starts_with(opcode, "LDMATRIX")) summary.ldmatrix++;
            if (is_cp_async_opcode(opcode)) summary.cp_async++;
            if (is_tensor_opcode(opcode)) summary.tensor_ops++;
        }
    }

    uint64_t shared_total = summary.shared_loads + summary.shared_stores;
    int score = 0;
    if (shared_total == 0) {
        summary.label = "no_shared";
        summary.risk_score = 0;
        return summary;
    }

    if (summary.tensor_ops > 0 && summary.ldmatrix == 0) score += 2;
    if (summary.cp_async == 0 && summary.shared_loads > 8) score += 1;
    if (summary.shared_stores * 2 > shared_total) score += 1;
    if (summary.vectorish_shared * 2 < shared_total) score += 1;
    if (summary.ldmatrix > 0) score -= 2;
    if (summary.cp_async > 0 || summary.global_to_shared > 0) score -= 1;
    if (score < 0) score = 0;
    if (score > 4) score = 4;
    summary.risk_score = score;
    if (score >= 3) summary.label = "high_suspicion";
    else if (score >= 1) summary.label = "medium_suspicion";
    else summary.label = "low_suspicion";
    return summary;
}

void nvbit_at_init() {
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    pthread_mutex_init(&launch_mutex, nullptr);
    if (const char* env = getenv("NVBPF_KERNEL_FILTER")) kernel_name_filter = env;
    verbose = getenv("NVBPF_VERBOSE") != nullptr;
    full_names = getenv("NVBPF_FULL_NAMES") != nullptr;
    printf("[NVBPF BANK_CONFLICT_SUSPICION] Tool loaded\n");
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (!is_exit || !nvbpf_is_launch_event(cbid)) return;
    CUfunction func = nvbpf_get_launch_func(cbid, params);
    const char* func_name = nvbit_get_func_name(ctx, func);
    if (!kernel_name_filter.empty() &&
        strstr(func_name, kernel_name_filter.c_str()) == nullptr) {
        return;
    }

    pthread_mutex_lock(&launch_mutex);
    matched_launches++;
    BankSummary* summary = find_summary(func_name);
    if (summary == nullptr) {
        summaries.push_back(analyze_bank_conflicts(ctx, func, func_name));
        summary = &summaries.back();
    }
    summary->launches++;

    if (verbose) {
        printf("[NVBPF] bank_conflict kernel=%s\n", func_name);
        printf("        shld=%lu shst=%lu ldmatrix=%lu cp_async=%lu tensor=%lu vectorish=%lu g2s=%lu score=%d (%s)\n",
               summary->shared_loads, summary->shared_stores, summary->ldmatrix,
               summary->cp_async, summary->tensor_ops, summary->vectorish_shared,
               summary->global_to_shared, summary->risk_score, summary->label.c_str());
    }
    pthread_mutex_unlock(&launch_mutex);
}

void nvbit_at_term() {
    if (!verbose) {
        printf("[NVBPF BANK_CONFLICT_SUSPICION] matched_launches=%lu unique_kernels=%zu\n",
               matched_launches, summaries.size());
        for (const auto& summary : summaries) {
            printf("  x%-3lu %-32s | shld=%-5lu shst=%-5lu ldm=%-4lu cp=%-4lu vec=%-4lu g2s=%-4lu score=%d | %s\n",
                   summary.launches,
                   compact_kernel_name(summary.kernel_name).c_str(),
                   summary.shared_loads,
                   summary.shared_stores,
                   summary.ldmatrix,
                   summary.cp_async,
                   summary.vectorish_shared,
                   summary.global_to_shared,
                   summary.risk_score,
                   summary.label.c_str());
        }
    }
    printf("[NVBPF BANK_CONFLICT_SUSPICION] Tool terminated\n");
}
