/*
 * NV-BPF Example: Epilogue Fusion Trace
 *
 * Host-side launch-neighborhood tool focused specifically on post-GEMM
 * epilogue behavior. This is narrower than gemm_orchestration_map: it only
 * asks whether useful post-processing appears fused into the focus kernel or
 * is still running as separate activation/bias/scale/copy kernels afterward.
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

enum EpilogueKind {
    EK_NONE = 0,
    EK_BIAS = 1,
    EK_ACTIVATION = 2,
    EK_SCALE = 3,
};

struct LaunchRecord {
    uint64_t event_id = 0;
    int gpu = -1;
    LaunchClass klass = LC_OTHER;
    bool is_kernel = false;
    size_t bytes = 0;
    std::string name;
};

struct FusionAggregate {
    std::string center_name;
    LaunchClass center_class = LC_OTHER;
    uint64_t count = 0;
    int post_window_min = 0, post_window_max = 0;
    int epi_min = 0, epi_max = 0;
    int bias_min = 0, bias_max = 0;
    int act_min = 0, act_max = 0;
    int scale_min = 0, scale_max = 0;
    int copy_min = 0, copy_max = 0;
    int red_min = 0, red_max = 0;
    int elem_min = 0, elem_max = 0;
    bool saw_fused_likely = false;
    bool saw_attention_core = false;
    bool saw_copyout = false;
    bool saw_reduction_tail = false;
    bool saw_separate_bias = false;
    bool saw_separate_activation = false;
    bool saw_separate_scale = false;
    bool saw_separate_generic = false;
};

static std::string filter_csv =
    "gemm,sgemm,hgemm,dgemm,bgemm,igemm,matmul,cublas,cutlass";
static int epilogue_window = 3;
static uint64_t api_event_counter = 0;
static bool verbose = false;
static bool full_names = false;
static std::vector<LaunchRecord> records;
static std::vector<FusionAggregate> aggregates;

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

static EpilogueKind classify_epilogue_kind(const char* name) {
    if (strstr(name, "bias")) return EK_BIAS;
    if (strstr(name, "relu") || strstr(name, "gelu") || strstr(name, "silu") ||
        strstr(name, "clamp") || strstr(name, "activation") ||
        strstr(name, "sigmoid") || strstr(name, "tanh")) {
        return EK_ACTIVATION;
    }
    if (strstr(name, "scale") || strstr(name, "mul") || strstr(name, "alpha") ||
        strstr(name, "beta")) {
        return EK_SCALE;
    }
    return EK_NONE;
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

static bool is_focus_kernel(const LaunchRecord& rec) {
    return rec.is_kernel &&
           (rec.klass == LC_GEMM || rec.klass == LC_ATTENTION ||
            csv_match(rec.name.c_str(), filter_csv));
}

static FusionAggregate* find_aggregate(const LaunchRecord& rec) {
    for (auto& agg : aggregates) {
        if (agg.center_class == rec.klass && agg.center_name == rec.name) {
            return &agg;
        }
    }
    return nullptr;
}

static void record_api_copy(const char* api_name, size_t bytes) {
    LaunchRecord rec{};
    rec.event_id = api_event_counter;
    rec.klass = LC_COPY;
    rec.bytes = bytes;
    rec.name = api_name;
    records.push_back(rec);
}

void nvbit_at_init() {
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    if (const char* env = getenv("NVBPF_GEMM_FILTER")) {
        filter_csv = env;
    }
    if (const char* env = getenv("NVBPF_EPILOGUE_WINDOW")) {
        epilogue_window = atoi(env);
        if (epilogue_window < 1) epilogue_window = 1;
        if (epilogue_window > 8) epilogue_window = 8;
    }
    verbose = getenv("NVBPF_VERBOSE") != nullptr;
    full_names = getenv("NVBPF_FULL_NAMES") != nullptr;
    printf("[NVBPF EPILOGUE_FUSION_TRACE] Tool loaded\n");
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
    size_t focus_launches = 0;
    size_t fused_likely = 0;
    size_t separate_launches = 0;

    for (size_t i = 0; i < records.size(); i++) {
        const auto& rec = records[i];
        if (!is_focus_kernel(rec)) continue;
        focus_launches++;

        size_t hi = i + (size_t)epilogue_window;
        if (hi >= records.size()) hi = records.size() - 1;
        for (size_t stop = i + 1; stop <= hi; stop++) {
            if (is_focus_kernel(records[stop])) {
                hi = stop - 1;
                break;
            }
        }

        int post_window = 0;
        int epi_after = 0;
        int bias_after = 0;
        int act_after = 0;
        int scale_after = 0;
        int copy_after = 0;
        int red_after = 0;
        int elem_after = 0;

        if (verbose) {
            printf("[NVBPF] epilogue_trace #%zu gpu=%d kernel=%s\n",
                   focus_launches, rec.gpu, rec.name.c_str());
        }

        for (size_t j = i + 1; j <= hi && j < records.size(); j++) {
            const auto& next = records[j];
            post_window++;
            if (verbose) {
                printf("         [+%ld] %-11s %s",
                       (long)j - (long)i, class_name(next.klass), next.name.c_str());
                if (!next.is_kernel && next.bytes > 0) {
                    printf(" bytes=%zu", next.bytes);
                }
                printf("\n");
            }

            if (next.klass == LC_COPY) copy_after++;
            if (next.klass == LC_REDUCTION) red_after++;
            if (next.klass == LC_ELEMENTWISE) elem_after++;
            if (next.klass == LC_EPILOGUE || next.klass == LC_ELEMENTWISE) {
                epi_after++;
                switch (classify_epilogue_kind(next.name.c_str())) {
                    case EK_BIAS: bias_after++; break;
                    case EK_ACTIVATION: act_after++; break;
                    case EK_SCALE: scale_after++; break;
                    default: break;
                }
            }
        }

        bool fused = (epi_after == 0 && copy_after == 0 && red_after == 0);
        if (fused) fused_likely++;
        if (!fused || bias_after > 0 || act_after > 0 || scale_after > 0) {
            separate_launches++;
        }

        if (verbose) {
            printf("        summary: post_window=%d epi=%d bias=%d act=%d scale=%d copy=%d red=%d elem=%d\n",
                   post_window, epi_after, bias_after, act_after, scale_after,
                   copy_after, red_after, elem_after);
            if (rec.klass == LC_ATTENTION) {
                printf("        heuristic: fused attention/FMHA kernel is the center of this trace\n");
            }
            if (fused) {
                printf("        heuristic: fused epilogue is likely; no obvious post-kernel epilogue/copy/reduction work nearby\n");
            } else {
                printf("        heuristic: post-kernel work suggests the epilogue is at least partially separate\n");
            }
            if (bias_after > 0) {
                printf("        heuristic: separate bias-like work detected after the focus kernel\n");
            }
            if (act_after > 0) {
                printf("        heuristic: separate activation-like work detected after the focus kernel\n");
            }
            if (scale_after > 0) {
                printf("        heuristic: separate scale-like work detected after the focus kernel\n");
            }
            if (copy_after > 0) {
                printf("        heuristic: explicit copy/reformat work follows the focus kernel\n");
            }
        } else {
            FusionAggregate* agg = find_aggregate(rec);
            if (agg == nullptr) {
                FusionAggregate fresh{};
                fresh.center_name = rec.name;
                fresh.center_class = rec.klass;
                fresh.post_window_min = fresh.post_window_max = post_window;
                fresh.epi_min = fresh.epi_max = epi_after;
                fresh.bias_min = fresh.bias_max = bias_after;
                fresh.act_min = fresh.act_max = act_after;
                fresh.scale_min = fresh.scale_max = scale_after;
                fresh.copy_min = fresh.copy_max = copy_after;
                fresh.red_min = fresh.red_max = red_after;
                fresh.elem_min = fresh.elem_max = elem_after;
                aggregates.push_back(fresh);
                agg = &aggregates.back();
            }
            agg->count++;
            update_range(post_window, &agg->post_window_min, &agg->post_window_max);
            update_range(epi_after, &agg->epi_min, &agg->epi_max);
            update_range(bias_after, &agg->bias_min, &agg->bias_max);
            update_range(act_after, &agg->act_min, &agg->act_max);
            update_range(scale_after, &agg->scale_min, &agg->scale_max);
            update_range(copy_after, &agg->copy_min, &agg->copy_max);
            update_range(red_after, &agg->red_min, &agg->red_max);
            update_range(elem_after, &agg->elem_min, &agg->elem_max);
            if (fused) agg->saw_fused_likely = true;
            if (rec.klass == LC_ATTENTION) agg->saw_attention_core = true;
            if (copy_after > 0) agg->saw_copyout = true;
            if (red_after > 0) agg->saw_reduction_tail = true;
            if (bias_after > 0) agg->saw_separate_bias = true;
            if (act_after > 0) agg->saw_separate_activation = true;
            if (scale_after > 0) agg->saw_separate_scale = true;
            if (epi_after > 0 && bias_after == 0 && act_after == 0 && scale_after == 0) {
                agg->saw_separate_generic = true;
            }
        }
    }

    printf("[NVBPF EPILOGUE_FUSION_TRACE] focus_kernels=%zu fused_likely=%zu separate_signals=%zu\n",
           focus_launches, fused_likely, separate_launches);

    if (!verbose) {
        printf("[NVBPF EPILOGUE_FUSION_TRACE] unique_focus_kernels=%zu\n", aggregates.size());
        for (const auto& agg : aggregates) {
            char post_buf[32], epi_buf[32], bias_buf[32], act_buf[32];
            char scale_buf[32], copy_buf[32], red_buf[32], elem_buf[32];
            format_range(post_buf, sizeof(post_buf), agg.post_window_min, agg.post_window_max);
            format_range(epi_buf, sizeof(epi_buf), agg.epi_min, agg.epi_max);
            format_range(bias_buf, sizeof(bias_buf), agg.bias_min, agg.bias_max);
            format_range(act_buf, sizeof(act_buf), agg.act_min, agg.act_max);
            format_range(scale_buf, sizeof(scale_buf), agg.scale_min, agg.scale_max);
            format_range(copy_buf, sizeof(copy_buf), agg.copy_min, agg.copy_max);
            format_range(red_buf, sizeof(red_buf), agg.red_min, agg.red_max);
            format_range(elem_buf, sizeof(elem_buf), agg.elem_min, agg.elem_max);

            printf("  x%-3lu %-10s %-32s | post=%s epi=%s bias=%s act=%s scale=%s copy=%s red=%s elem=%s | ",
                   agg.count,
                   class_name(agg.center_class),
                   compact_kernel_name(agg.center_name).c_str(),
                   post_buf, epi_buf, bias_buf, act_buf, scale_buf, copy_buf,
                   red_buf, elem_buf);
            bool wrote = false;
            if (agg.saw_attention_core) {
                printf("attention_core");
                wrote = true;
            }
            if (agg.saw_fused_likely) {
                printf("%sfused_likely", wrote ? "," : "");
                wrote = true;
            }
            if (agg.saw_separate_bias) {
                printf("%sseparate_bias", wrote ? "," : "");
                wrote = true;
            }
            if (agg.saw_separate_activation) {
                printf("%sseparate_activation", wrote ? "," : "");
                wrote = true;
            }
            if (agg.saw_separate_scale) {
                printf("%sseparate_scale", wrote ? "," : "");
                wrote = true;
            }
            if (agg.saw_separate_generic) {
                printf("%sseparate_epilogue", wrote ? "," : "");
                wrote = true;
            }
            if (agg.saw_copyout) {
                printf("%scopyout_after", wrote ? "," : "");
                wrote = true;
            }
            if (agg.saw_reduction_tail) {
                printf("%sreduction_tail", wrote ? "," : "");
                wrote = true;
            }
            if (!wrote) {
                printf("no_clear_signal");
            }
            printf("\n");
        }
    }

    printf("[NVBPF EPILOGUE_FUSION_TRACE] Tool terminated\n");
}
