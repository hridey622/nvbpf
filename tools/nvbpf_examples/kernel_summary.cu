/*
 * NV-BPF Example: Kernel Summary
 *
 * ML-oriented per-kernel summary with instruction mix, memory mix,
 * launch geometry, and active-SM distribution.
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
BPF_ARRAY(tensor_instrs, uint64_t, 1);
BPF_ARRAY(ffma_instrs, uint64_t, 1);
BPF_ARRAY(ldmatrix_instrs, uint64_t, 1);
BPF_ARRAY(cp_async_instrs, uint64_t, 1);
BPF_ARRAY(branch_instrs, uint64_t, 1);
BPF_ARRAY(load_instrs, uint64_t, 1);
BPF_ARRAY(store_instrs, uint64_t, 1);
BPF_PERCPU_ARRAY(sm_instrs, uint64_t, 1);
BPF_ARRAY(active_sm_bitmap, uint64_t, 4);

extern "C" __device__ __noinline__ void ks_count_instr(int pred,
                                                       uint64_t ptotal,
                                                       uint64_t psm_instrs,
                                                       uint64_t pbitmap);
extern "C" __device__ __noinline__ void ks_count_counter(int pred,
                                                         uint64_t pcounter);

static pthread_mutex_t launch_mutex;
static std::unordered_set<CUfunction> already_instrumented;
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

static const char* classify_kernel(const char* name) {
    if (strstr(name, "attention") || strstr(name, "attn") || strstr(name, "softmax")) {
        return "attention";
    }
    if (strstr(name, "gemm") || strstr(name, "sgemm") || strstr(name, "bmm")) {
        return "gemm";
    }
    if (strstr(name, "nccl") || strstr(name, "allreduce") || strstr(name, "sendrecv")) {
        return "comm";
    }
    if (strstr(name, "copy") || strstr(name, "cast")) {
        return "copy";
    }
    if (strstr(name, "elementwise") || strstr(name, "vectorized")) {
        return "elementwise";
    }
    return "other";
}

static void reset_state() {
    total_instrs.reset();
    tensor_instrs.reset();
    ffma_instrs.reset();
    ldmatrix_instrs.reset();
    cp_async_instrs.reset();
    branch_instrs.reset();
    load_instrs.reset();
    store_instrs.reset();
    sm_instrs.reset();
    active_sm_bitmap.reset();
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

            nvbit_insert_call(instr, "ks_count_instr", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&total_instrs.data[0]);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&sm_instrs.data[0][0]);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&active_sm_bitmap.data[0]);

            if (is_tensor_opcode(opcode)) {
                nvbit_insert_call(instr, "ks_count_counter", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&tensor_instrs.data[0]);
            }
            if (opcode_starts_with(opcode, "FFMA")) {
                nvbit_insert_call(instr, "ks_count_counter", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&ffma_instrs.data[0]);
            }
            if (opcode_starts_with(opcode, "LDMATRIX")) {
                nvbit_insert_call(instr, "ks_count_counter", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&ldmatrix_instrs.data[0]);
            }
            if (is_cp_async_opcode(opcode)) {
                nvbit_insert_call(instr, "ks_count_counter", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&cp_async_instrs.data[0]);
            }
            if (is_branch_opcode(opcode)) {
                nvbit_insert_call(instr, "ks_count_counter", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&branch_instrs.data[0]);
            }
            if (instr->isLoad() &&
                instr->getMemorySpace() != InstrType::MemorySpace::CONSTANT) {
                nvbit_insert_call(instr, "ks_count_counter", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&load_instrs.data[0]);
            }
            if (instr->isStore()) {
                nvbit_insert_call(instr, "ks_count_counter", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)&store_instrs.data[0]);
            }
        }
    }
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
    if (const char* env = getenv("NVBPF_KERNEL_FILTER")) {
        kernel_name_filter = env;
    }
    printf("[NVBPF KERNEL_SUMMARY] Tool loaded\n");
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
        if (!match) {
            pthread_mutex_unlock(&launch_mutex);
        }
    } else {
        cudaDeviceSynchronize();
        if (!kernel_name_filter.empty() &&
            strstr(func_name, kernel_name_filter.c_str()) == nullptr) {
            return;
        }

        func_config_t cfg{};
        nvbit_get_func_config(ctx, func, &cfg);

        int gx, gy, gz, bx, by, bz;
        launch_dims(cbid, params, &gx, &gy, &gz, &bx, &by, &bz);

        uint64_t total = *total_instrs.lookup(0);
        uint64_t tensor = *tensor_instrs.lookup(0);
        uint64_t ffma = *ffma_instrs.lookup(0);
        uint64_t ldmatrix = *ldmatrix_instrs.lookup(0);
        uint64_t cp_async = *cp_async_instrs.lookup(0);
        uint64_t branches = *branch_instrs.lookup(0);
        uint64_t loads = *load_instrs.lookup(0);
        uint64_t stores = *store_instrs.lookup(0);

        int active_sms = 0;
        for (int word = 0; word < 4; word++) {
            uint64_t* bm = active_sm_bitmap.lookup(word);
            if (bm) active_sms += __builtin_popcountll(*bm);
        }

        printf("[NVBPF] %s (%s)\n", func_name, classify_kernel(func_name));
        printf("        launch: grid=(%d,%d,%d) block=(%d,%d,%d) regs=%u smem=%u+%u\n",
               gx, gy, gz, bx, by, bz, cfg.num_registers,
               cfg.shmem_static_nbytes, cfg.shmem_dynamic_nbytes);
        printf("        instrs=%lu tensor=%lu ffma=%lu ldmatrix=%lu cp_async=%lu branches=%lu\n",
               total, tensor, ffma, ldmatrix, cp_async, branches);
        printf("        mem: loads=%lu stores=%lu active_sms=%d\n",
               loads, stores, active_sms);
        pthread_mutex_unlock(&launch_mutex);
    }
}

void nvbit_at_term() {
    printf("[NVBPF KERNEL_SUMMARY] Tool terminated\n");
}
