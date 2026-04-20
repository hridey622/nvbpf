/*
 * NV-BPF Example: GEMM Orchestration Map
 *
 * Host-side launch neighborhood trace for GEMM-like kernels. This is intended
 * to explain whether a GEMM appears surrounded by prep/copy/epilogue kernels
 * rather than focusing only on the main compute kernel in isolation.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

enum LaunchClass {
    LC_ATTENTION = 0,
    LC_GEMM = 1,
    LC_EPILOGUE = 2,
    LC_COPY = 3,
    LC_TRANSPOSE = 4,
    LC_REDUCTION = 5,
    LC_ELEMENTWISE = 6,
    LC_OTHER = 7,
};

struct LaunchRecord {
    uint64_t event_id = 0;
    int gpu = -1;
    LaunchClass klass = LC_OTHER;
    bool is_kernel = false;
    size_t bytes = 0;
    std::string name;
};

static std::string filter_csv =
    "gemm,sgemm,hgemm,dgemm,bgemm,igemm,matmul,cublas,cutlass";
static int neighborhood_window = 2;
static uint64_t api_event_counter = 0;
static std::vector<LaunchRecord> records;
static bool verbose = false;
static bool full_names = false;

struct NeighborhoodAggregate {
    std::string center_name;
    LaunchClass center_class = LC_OTHER;
    uint64_t count = 0;
    int prep_min = 0, prep_max = 0;
    int copy_min = 0, copy_max = 0;
    int transpose_min = 0, transpose_max = 0;
    int epilogue_min = 0, epilogue_max = 0;
    int elementwise_min = 0, elementwise_max = 0;
    int reduction_min = 0, reduction_max = 0;
    int attention_min = 0, attention_max = 0;
    bool saw_prep = false;
    bool saw_fused_attention = false;
    bool saw_separate_epilogue = false;
};

static std::vector<NeighborhoodAggregate> aggregates;

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

static void update_range(int value, int* min_value, int* max_value) {
    if (value < *min_value) *min_value = value;
    if (value > *max_value) *max_value = value;
}

static void format_range(char* out, size_t out_size, int min_value, int max_value) {
    if (min_value == max_value) {
        snprintf(out, out_size, "%d", min_value);
    } else {
        snprintf(out, out_size, "%d-%d", min_value, max_value);
    }
}

static NeighborhoodAggregate* find_aggregate(const LaunchRecord& rec) {
    for (auto& agg : aggregates) {
        if (agg.center_class == rec.klass && agg.center_name == rec.name) {
            return &agg;
        }
    }
    return nullptr;
}

static LaunchClass classify_name(const char* name) {
    if (strstr(name, "fmha") || strstr(name, "flash") || strstr(name, "attention") ||
        strstr(name, "attn")) {
        return LC_ATTENTION;
    }
    if (strstr(name, "gemm") || strstr(name, "sgemm") || strstr(name, "matmul") ||
        strstr(name, "cublas") || strstr(name, "cutlass") || strstr(name, "wmma")) {
        return LC_GEMM;
    }
    if (strstr(name, "epilogue") || strstr(name, "bias") || strstr(name, "relu") ||
        strstr(name, "gelu") || strstr(name, "silu") || strstr(name, "clamp") ||
        strstr(name, "activation")) {
        return LC_EPILOGUE;
    }
    if (strstr(name, "copy") || strstr(name, "cast") || strstr(name, "memcpy") ||
        strstr(name, "reformat") || strstr(name, "convert")) {
        return LC_COPY;
    }
    if (strstr(name, "transpose") || strstr(name, "permute") ||
        strstr(name, "layout")) {
        return LC_TRANSPOSE;
    }
    if (strstr(name, "reduce") || strstr(name, "reduction") ||
        strstr(name, "softmax") || strstr(name, "layernorm")) {
        return LC_REDUCTION;
    }
    if (strstr(name, "elementwise") || strstr(name, "vectorized") ||
        strstr(name, "unrolled_elementwise") || strstr(name, "binary") ||
        strstr(name, "unary") || strstr(name, "mul") || strstr(name, "add")) {
        return LC_ELEMENTWISE;
    }
    return LC_OTHER;
}

static const char* class_name(LaunchClass klass) {
    switch (klass) {
        case LC_ATTENTION: return "attention";
        case LC_GEMM: return "gemm";
        case LC_EPILOGUE: return "epilogue";
        case LC_COPY: return "copy";
        case LC_TRANSPOSE: return "transpose";
        case LC_REDUCTION: return "reduction";
        case LC_ELEMENTWISE: return "elementwise";
        default: return "other";
    }
}

static void record_api_copy(const char* api_name, size_t bytes) {
    LaunchRecord rec{};
    rec.event_id = api_event_counter;
    rec.klass = LC_COPY;
    rec.bytes = bytes;
    rec.name = api_name;
    records.push_back(rec);
}

static bool is_focus_gemm(const LaunchRecord& rec) {
    return rec.is_kernel &&
           (rec.klass == LC_GEMM || rec.klass == LC_ATTENTION ||
            csv_match(rec.name.c_str(), filter_csv));
}

void nvbit_at_init() {
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    if (const char* env = getenv("NVBPF_GEMM_FILTER")) {
        filter_csv = env;
    }
    if (const char* env = getenv("NVBPF_ORCH_WINDOW")) {
        neighborhood_window = atoi(env);
        if (neighborhood_window < 1) neighborhood_window = 1;
        if (neighborhood_window > 8) neighborhood_window = 8;
    }
    verbose = getenv("NVBPF_VERBOSE") != nullptr;
    full_names = getenv("NVBPF_FULL_NAMES") != nullptr;
    printf("[NVBPF GEMM_ORCHESTRATION_MAP] Tool loaded\n");
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (!is_exit) return;
    api_event_counter++;

    if (cbid == API_CUDA_cuMemcpyPeer || cbid == API_CUDA_cuMemcpyPeer_ptds) {
        size_t bytes = 0;
        if (cbid == API_CUDA_cuMemcpyPeer) {
            bytes = ((cuMemcpyPeer_params*)params)->ByteCount;
        } else {
            bytes = ((cuMemcpyPeer_ptds_params*)params)->ByteCount;
        }
        record_api_copy("api:cuMemcpyPeer", bytes);
        return;
    }
    if (cbid == API_CUDA_cuMemcpyPeerAsync || cbid == API_CUDA_cuMemcpyPeerAsync_ptsz) {
        size_t bytes = 0;
        if (cbid == API_CUDA_cuMemcpyPeerAsync) {
            bytes = ((cuMemcpyPeerAsync_params*)params)->ByteCount;
        } else {
            bytes = ((cuMemcpyPeerAsync_ptsz_params*)params)->ByteCount;
        }
        record_api_copy("api:cuMemcpyPeerAsync", bytes);
        return;
    }
    if (cbid == API_CUDA_cuMemcpyDtoD_v2 ||
        cbid == API_CUDA_cuMemcpyDtoDAsync_v2 ||
        cbid == API_CUDA_cuMemcpyDtoD_v2_ptds ||
        cbid == API_CUDA_cuMemcpyDtoDAsync_v2_ptsz) {
        size_t bytes = 0;
        const char* label = "api:cuMemcpyDtoD";
        if (cbid == API_CUDA_cuMemcpyDtoD_v2) {
            bytes = ((cuMemcpyDtoD_v2_params*)params)->ByteCount;
        } else if (cbid == API_CUDA_cuMemcpyDtoDAsync_v2) {
            bytes = ((cuMemcpyDtoDAsync_v2_params*)params)->ByteCount;
            label = "api:cuMemcpyDtoDAsync";
        } else if (cbid == API_CUDA_cuMemcpyDtoD_v2_ptds) {
            bytes = ((cuMemcpyDtoD_v2_ptds_params*)params)->ByteCount;
        } else {
            bytes = ((cuMemcpyDtoDAsync_v2_ptsz_params*)params)->ByteCount;
            label = "api:cuMemcpyDtoDAsync";
        }
        record_api_copy(label, bytes);
        return;
    }

    if (!nvbpf_is_launch_event(cbid)) return;

    CUfunction func = nvbpf_get_launch_func(cbid, params);
    const char* func_name = nvbit_get_func_name(ctx, func);

    LaunchRecord rec{};
    rec.event_id = api_event_counter;
    rec.is_kernel = true;
    rec.klass = classify_name(func_name);
    rec.name = func_name;
    CUdevice dev = 0;
    if (cuCtxGetDevice(&dev) == CUDA_SUCCESS) {
        rec.gpu = (int)dev;
    }
    records.push_back(rec);
}

void nvbit_at_term() {
    size_t kernel_launches = 0;
    size_t gemm_launches = 0;
    size_t copy_events = 0;
    for (const auto& rec : records) {
        if (rec.is_kernel) kernel_launches++;
        if (is_focus_gemm(rec)) gemm_launches++;
        if (rec.klass == LC_COPY) copy_events++;
    }

    printf("[NVBPF GEMM_ORCHESTRATION_MAP] launches=%zu focus_kernels=%zu copy_events=%zu\n",
           kernel_launches, gemm_launches, copy_events);

    size_t ordinal = 0;
    for (size_t i = 0; i < records.size(); i++) {
        const auto& rec = records[i];
        if (!is_focus_gemm(rec)) continue;
        ordinal++;

        int prep_before = 0;
        int copy_before = 0;
        int epilogue_after = 0;
        int transpose_before = 0;
        int elementwise_after = 0;
        int reduction_after = 0;
        int attention_neighbors = 0;

        size_t lo = (i > (size_t)neighborhood_window) ? i - neighborhood_window : 0;
        size_t hi = i + neighborhood_window;
        if (hi >= records.size()) hi = records.size() - 1;

        if (verbose) {
            printf("[NVBPF] gemm_neighborhood #%zu gpu=%d kernel=%s\n",
                   ordinal, rec.gpu, rec.name.c_str());
        }
        for (size_t j = lo; j <= hi; j++) {
            if (j == i) {
                if (verbose) {
                    printf("          [0] %-11s %s\n", class_name(records[j].klass),
                           records[j].name.c_str());
                }
                continue;
            }
            if (verbose) {
                long rel = (long)j - (long)i;
                printf("         [%+ld] %-11s %s",
                       rel, class_name(records[j].klass), records[j].name.c_str());
                if (!records[j].is_kernel && records[j].bytes > 0) {
                    printf(" bytes=%zu", records[j].bytes);
                }
                printf("\n");
            }

            if (j < i) {
                if (records[j].klass == LC_COPY) copy_before++;
                if (records[j].klass == LC_TRANSPOSE) transpose_before++;
                if (records[j].klass == LC_COPY || records[j].klass == LC_TRANSPOSE) {
                    prep_before++;
                }
            } else if (j > i) {
                if (records[j].klass == LC_EPILOGUE) epilogue_after++;
                if (records[j].klass == LC_ELEMENTWISE) elementwise_after++;
                if (records[j].klass == LC_REDUCTION) reduction_after++;
            }
            if (records[j].klass == LC_ATTENTION) attention_neighbors++;
        }

        if (verbose) {
            printf("        summary: prep_before=%d copy_before=%d transpose_before=%d epilogue_after=%d elementwise_after=%d reduction_after=%d attention_neighbors=%d\n",
                   prep_before, copy_before, transpose_before, epilogue_after,
                   elementwise_after, reduction_after, attention_neighbors);
            if (prep_before > 0) {
                printf("        heuristic: GEMM is surrounded by explicit prep/copy work\n");
            }
            if (rec.klass == LC_ATTENTION || attention_neighbors > 0) {
                printf("        heuristic: fused attention/FMHA kernels are present in the local neighborhood\n");
            }
            if (epilogue_after > 0 || elementwise_after > 0) {
                printf("        heuristic: post-GEMM epilogue appears to run as separate kernels\n");
            } else {
                printf("        heuristic: no obvious post-GEMM epilogue kernels in the local neighborhood; fused epilogue is possible\n");
            }
        } else {
            NeighborhoodAggregate* agg = find_aggregate(rec);
            if (agg == nullptr) {
                NeighborhoodAggregate fresh{};
                fresh.center_name = rec.name;
                fresh.center_class = rec.klass;
                fresh.prep_min = fresh.prep_max = prep_before;
                fresh.copy_min = fresh.copy_max = copy_before;
                fresh.transpose_min = fresh.transpose_max = transpose_before;
                fresh.epilogue_min = fresh.epilogue_max = epilogue_after;
                fresh.elementwise_min = fresh.elementwise_max = elementwise_after;
                fresh.reduction_min = fresh.reduction_max = reduction_after;
                fresh.attention_min = fresh.attention_max = attention_neighbors;
                aggregates.push_back(fresh);
                agg = &aggregates.back();
            }
            agg->count++;
            update_range(prep_before, &agg->prep_min, &agg->prep_max);
            update_range(copy_before, &agg->copy_min, &agg->copy_max);
            update_range(transpose_before, &agg->transpose_min, &agg->transpose_max);
            update_range(epilogue_after, &agg->epilogue_min, &agg->epilogue_max);
            update_range(elementwise_after, &agg->elementwise_min, &agg->elementwise_max);
            update_range(reduction_after, &agg->reduction_min, &agg->reduction_max);
            update_range(attention_neighbors, &agg->attention_min, &agg->attention_max);
            if (prep_before > 0) agg->saw_prep = true;
            if (rec.klass == LC_ATTENTION || attention_neighbors > 0) agg->saw_fused_attention = true;
            if (epilogue_after > 0 || elementwise_after > 0) agg->saw_separate_epilogue = true;
        }
    }

    if (!verbose) {
        printf("[NVBPF GEMM_ORCHESTRATION_MAP] unique_focus_kernels=%zu\n", aggregates.size());
        for (const auto& agg : aggregates) {
            char prep_buf[32], copy_buf[32], trans_buf[32], epi_buf[32];
            char elem_buf[32], red_buf[32], attn_buf[32];
            format_range(prep_buf, sizeof(prep_buf), agg.prep_min, agg.prep_max);
            format_range(copy_buf, sizeof(copy_buf), agg.copy_min, agg.copy_max);
            format_range(trans_buf, sizeof(trans_buf), agg.transpose_min, agg.transpose_max);
            format_range(epi_buf, sizeof(epi_buf), agg.epilogue_min, agg.epilogue_max);
            format_range(elem_buf, sizeof(elem_buf), agg.elementwise_min, agg.elementwise_max);
            format_range(red_buf, sizeof(red_buf), agg.reduction_min, agg.reduction_max);
            format_range(attn_buf, sizeof(attn_buf), agg.attention_min, agg.attention_max);
            printf("  x%-3lu %-10s %-32s | prep=%s copy=%s trans=%s epi=%s elem=%s red=%s attn=%s | ",
                   agg.count,
                   class_name(agg.center_class),
                   compact_kernel_name(agg.center_name).c_str(),
                   prep_buf, copy_buf, trans_buf, epi_buf, elem_buf, red_buf, attn_buf);
            bool wrote = false;
            if (agg.saw_fused_attention) {
                printf("fused_attention");
                wrote = true;
            }
            if (agg.saw_prep) {
                printf("%sprep_copy", wrote ? "," : "");
                wrote = true;
            }
            if (agg.saw_separate_epilogue) {
                printf("%sseparate_epilogue", wrote ? "," : "");
                wrote = true;
            }
            if (!wrote) {
                printf("clean_neighborhood");
            }
            printf("\n");
        }
    }

    printf("[NVBPF GEMM_ORCHESTRATION_MAP] Tool terminated\n");
}
