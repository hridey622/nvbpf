/*
 * NV-BPF Example: GEMM Wavefit Trace
 *
 * Estimates CTA wave fit for GEMM-like kernels and shows how evenly CTAs
 * are distributed across SMs. This is meant to answer "does this launch
 * quantize well into resident CTA waves?" rather than traditional occupancy.
 */

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <unordered_set>
#include <vector>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

BPF_PERCPU_ARRAY(sm_cta_entries, uint64_t, 1);
BPF_ARRAY(active_sm_bitmap, uint64_t, 4);

extern "C" __device__ __noinline__ void gwt_kernel_entry(int pred,
                                                         uint64_t psm_entries,
                                                         uint64_t pbitmap);

static pthread_mutex_t launch_mutex;
static std::unordered_set<CUfunction> already_instrumented;
static std::string filter_csv =
    "gemm,sgemm,hgemm,dgemm,bgemm,igemm,matmul,cublas,cutlass";
static bool verbose = false;
static bool full_names = false;
static uint64_t matched_launches = 0;

struct WavefitSummary {
    std::string kernel_name;
    uint64_t launches = 0;
    int gx = 1, gy = 1, gz = 1;
    int bx = 1, by = 1, bz = 1;
    uint64_t total_ctas = 0;
    uint32_t regs = 0;
    uint32_t smem_static = 0;
    uint32_t smem_dynamic = 0;
    int sm_count = 0;
    int resident_ctas_per_sm = 1;
    uint64_t wave_capacity = 0;
    double fill_fraction = 0.0;
    int active_sms = 0;
    int used_sms = 0;
    uint64_t tail_empty_slots = 0;
    int heuristic_code = 0;
};

static std::vector<WavefitSummary> summaries;

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

static const char* heuristic_label(int code) {
    switch (code) {
        case 0: return "underfill";
        case 1: return "perfect_fit";
        case 2: return "small_tail";
        default: return "large_tail";
    }
}

static WavefitSummary* find_summary(const char* func_name,
                                    int gx, int gy, int gz,
                                    int bx, int by, int bz,
                                    uint32_t regs,
                                    uint32_t smem_static,
                                    uint32_t smem_dynamic) {
    for (auto& summary : summaries) {
        if (summary.kernel_name == func_name &&
            summary.gx == gx && summary.gy == gy && summary.gz == gz &&
            summary.bx == bx && summary.by == by && summary.bz == bz &&
            summary.regs == regs &&
            summary.smem_static == smem_static &&
            summary.smem_dynamic == smem_dynamic) {
            return &summary;
        }
    }
    return nullptr;
}

static void reset_state() {
    sm_cta_entries.reset();
    active_sm_bitmap.reset();
}

static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    if (!already_instrumented.insert(func).second) {
        return;
    }

    const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, func);
    if (instrs.empty()) {
        return;
    }

    nvbit_insert_call(instrs[0], "gwt_kernel_entry", IPOINT_BEFORE);
    nvbit_add_call_arg_guard_pred_val(instrs[0]);
    nvbit_add_call_arg_const_val64(instrs[0], (uint64_t)&sm_cta_entries.data[0][0]);
    nvbit_add_call_arg_const_val64(instrs[0], (uint64_t)&active_sm_bitmap.data[0]);
}

static void launch_dims(nvbit_api_cuda_t cbid, void* params,
                        int* gx, int* gy, int* gz,
                        int* bx, int* by, int* bz) {
    *gx = *gy = *gz = *bx = *by = *bz = 1;
    if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
        cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
        *gx = p->config->gridDimX;
        *gy = p->config->gridDimY;
        *gz = p->config->gridDimZ;
        *bx = p->config->blockDimX;
        *by = p->config->blockDimY;
        *bz = p->config->blockDimZ;
    } else if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel) {
        cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
        *gx = p->gridDimX;
        *gy = p->gridDimY;
        *gz = p->gridDimZ;
        *bx = p->blockDimX;
        *by = p->blockDimY;
        *bz = p->blockDimZ;
    }
}

void nvbit_at_init() {
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    pthread_mutex_init(&launch_mutex, nullptr);
    if (const char* env = getenv("NVBPF_GEMM_FILTER")) {
        filter_csv = env;
    }
    verbose = getenv("NVBPF_VERBOSE") != nullptr;
    full_names = getenv("NVBPF_FULL_NAMES") != nullptr;
    printf("[NVBPF GEMM_WAVEFIT_TRACE] Tool loaded\n");
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
        if (!match) {
            pthread_mutex_unlock(&launch_mutex);
        }
    } else {
        cudaDeviceSynchronize();
        if (!csv_match(func_name, filter_csv)) {
            return;
        }
        matched_launches++;

        int gx, gy, gz, bx, by, bz;
        launch_dims(cbid, params, &gx, &gy, &gz, &bx, &by, &bz);
        uint64_t total_ctas = (uint64_t)gx * gy * gz;

        CUdevice dev = 0;
        cuCtxGetDevice(&dev);

        int sm_count = 0;
        cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);

        func_config_t cfg{};
        nvbit_get_func_config(ctx, func, &cfg);

        int resident_ctas_per_sm = 1;
        int threads_per_block = bx * by * bz;
        if (threads_per_block > 0) {
            CUresult occ_status = cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &resident_ctas_per_sm, func, threads_per_block, cfg.shmem_dynamic_nbytes);
            if (occ_status != CUDA_SUCCESS || resident_ctas_per_sm <= 0) {
                resident_ctas_per_sm = 1;
            }
        }

        uint64_t wave_capacity = (uint64_t)sm_count * resident_ctas_per_sm;
        uint64_t full_waves = wave_capacity > 0 ? total_ctas / wave_capacity : 0;
        uint64_t tail_ctas = wave_capacity > 0 ? total_ctas % wave_capacity : 0;
        uint64_t tail_empty_slots = (tail_ctas > 0 && wave_capacity > tail_ctas)
                                        ? wave_capacity - tail_ctas
                                        : 0;
        double tail_fraction =
            (tail_ctas > 0 && wave_capacity > 0)
                ? (double)tail_ctas / (double)wave_capacity
                : 0.0;
        double launch_fill_fraction =
            (wave_capacity > 0)
                ? (double)total_ctas / (double)wave_capacity
                : 0.0;

        int active_sms = 0;
        for (int word = 0; word < 4; word++) {
            uint64_t* bm = active_sm_bitmap.lookup(word);
            if (bm) active_sms += __builtin_popcountll(*bm);
        }

        uint64_t sum_entries = 0;
        uint64_t min_entries = UINT64_MAX;
        uint64_t max_entries = 0;
        int used_sms = 0;
        for (int sm = 0; sm < sm_count; sm++) {
            uint64_t* entries = sm_cta_entries.lookup_sm(sm, 0);
            uint64_t value = entries ? *entries : 0;
            if (value == 0) continue;
            used_sms++;
            sum_entries += value;
            if (value < min_entries) min_entries = value;
            if (value > max_entries) max_entries = value;
        }
        if (min_entries == UINT64_MAX) min_entries = 0;

        double avg_entries =
            used_sms > 0 ? (double)sum_entries / (double)used_sms : 0.0;

        int heuristic_code = 0;
        if (full_waves == 0 && tail_ctas > 0) {
            heuristic_code = 0;
        } else if (tail_ctas == 0) {
            heuristic_code = 1;
        } else if (tail_fraction < 0.25) {
            heuristic_code = 2;
        } else {
            heuristic_code = 3;
        }

        if (verbose) {
            printf("[NVBPF] gemm_wavefit kernel=%s\n", func_name);
            printf("        launch: grid=(%d,%d,%d) block=(%d,%d,%d) total_ctas=%lu regs=%u smem=%u+%u\n",
                   gx, gy, gz, bx, by, bz, total_ctas, cfg.num_registers,
                   cfg.shmem_static_nbytes, cfg.shmem_dynamic_nbytes);
            printf("        waves: sms=%d resident_ctas_per_sm=%d wave_capacity=%lu full_waves=%lu tail_ctas=%lu tail_fraction=%.3f fill_fraction=%.3f\n",
                   sm_count, resident_ctas_per_sm, wave_capacity, full_waves,
                   tail_ctas, tail_fraction, launch_fill_fraction);
            printf("        distribution: active_sms=%d used_sms=%d cta_entries=%lu min=%lu avg=%.2f max=%lu tail_empty_slots=%lu\n",
                   active_sms, used_sms, sum_entries, min_entries, avg_entries, max_entries,
                   tail_empty_slots);
            if (heuristic_code == 0) {
                printf("        heuristic: launch does not fill a single resident wave; severe wave underfill dominates utilization\n");
            } else if (heuristic_code == 1) {
                printf("        heuristic: perfect wave fit for estimated resident CTA capacity\n");
            } else if (heuristic_code == 2) {
                printf("        heuristic: small tail wave; underfill is limited\n");
            } else {
                printf("        heuristic: sizable partial tail wave; launch shape likely wastes a noticeable final wave\n");
            }
        } else {
            WavefitSummary* summary = find_summary(
                func_name, gx, gy, gz, bx, by, bz, cfg.num_registers,
                cfg.shmem_static_nbytes, cfg.shmem_dynamic_nbytes);
            if (summary == nullptr) {
                WavefitSummary fresh{};
                fresh.kernel_name = func_name;
                fresh.gx = gx; fresh.gy = gy; fresh.gz = gz;
                fresh.bx = bx; fresh.by = by; fresh.bz = bz;
                fresh.total_ctas = total_ctas;
                fresh.regs = cfg.num_registers;
                fresh.smem_static = cfg.shmem_static_nbytes;
                fresh.smem_dynamic = cfg.shmem_dynamic_nbytes;
                fresh.sm_count = sm_count;
                fresh.resident_ctas_per_sm = resident_ctas_per_sm;
                fresh.wave_capacity = wave_capacity;
                fresh.fill_fraction = launch_fill_fraction;
                fresh.active_sms = active_sms;
                fresh.used_sms = used_sms;
                fresh.tail_empty_slots = tail_empty_slots;
                fresh.heuristic_code = heuristic_code;
                summaries.push_back(fresh);
                summary = &summaries.back();
            }
            summary->launches++;
        }
        if (used_sms > 0 && max_entries > min_entries + 1) {
            if (verbose) {
                printf("        heuristic: CTA distribution is uneven across active SMs\n");
            }
        }
        pthread_mutex_unlock(&launch_mutex);
    }
}

void nvbit_at_term() {
    if (!verbose) {
        printf("[NVBPF GEMM_WAVEFIT_TRACE] matched_launches=%lu unique_kernels=%zu\n",
               matched_launches, summaries.size());
        for (const auto& summary : summaries) {
            printf("  x%-3lu %-32s | ctas=%-4lu fill=%.3f sms=%d/%d regs=%u smem=%u+%u | %s\n",
                   summary.launches,
                   compact_kernel_name(summary.kernel_name).c_str(),
                   summary.total_ctas,
                   summary.fill_fraction,
                   summary.used_sms,
                   summary.sm_count,
                   summary.regs,
                   summary.smem_static,
                   summary.smem_dynamic,
                   heuristic_label(summary.heuristic_code));
        }
    }
    printf("[NVBPF GEMM_WAVEFIT_TRACE] Tool terminated\n");
}
