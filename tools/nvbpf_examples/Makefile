# NV-BPF Examples Makefile
# SPDX-License-Identifier: BSD-3-Clause

NVCC=nvcc -ccbin=$(CXX) -D_FORCE_INLINES
PTXAS=ptxas

NVCC_VER_REQ=10.1
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${NVCC_VER} >= $(NVCC_VER_REQ)" | bc)

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool!)
endif

PTXAS_VER_ADD_FLAG=12.3
PTXAS_VER=$(shell $(PTXAS) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
PTXAS_VER_CHECK=$(shell echo "${PTXAS_VER} >= $(PTXAS_VER_ADD_FLAG)" | bc)

ifeq ($(PTXAS_VER_CHECK), 0)
MAXRREGCOUNT_FLAG=-maxrregcount=24
else
MAXRREGCOUNT_FLAG=
endif

NVBIT_PATH=../../core
INCLUDES=-I$(NVBIT_PATH)

LIBS=-L$(NVBIT_PATH) -lnvbit
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

ARCH?=sm_89

# Flags for host-side tool files (NVBit callbacks, maps, hook registration)
HOST_FLAGS=-dc -c -std=c++17 $(INCLUDES) -Xptxas -cloning=no \
           -Xcompiler -Wall -arch=$(ARCH) -O3 -Xcompiler -fPIC

# Flags for device-side hook files (must keplesep device functions for NVBit injection)
HOOK_FLAGS=$(INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch \
           --keep-device-functions -arch=$(ARCH) \
           -Xcompiler -Wall -Xcompiler -fPIC -c

all: instr_count.so mem_trace.so sm_profiler.so attention_debug.so \
     kernel_summary.so attention_trace.so sampling_mem_trace.so \
     branch_divergence.so nvlink_trace.so

# ── instr_count ──────────────────────────────────────────────────────────────
instr_count.so: instr_count.o instr_count_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 instr_count.o instr_count_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

instr_count.o: instr_count.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

instr_count_hooks.o: instr_count_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── mem_trace ────────────────────────────────────────────────────────────────
mem_trace.so: mem_trace.o mem_trace_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 mem_trace.o mem_trace_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

mem_trace.o: mem_trace.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

mem_trace_hooks.o: mem_trace_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── sm_profiler ───────────────────────────────────────────────────────────────
sm_profiler.so: sm_profiler.o sm_profiler_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 sm_profiler.o sm_profiler_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

sm_profiler.o: sm_profiler.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

sm_profiler_hooks.o: sm_profiler_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── attention_debug ───────────────────────────────────────────────────────────
attention_debug.so: attention_debug.o attention_debug_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 attention_debug.o attention_debug_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@
attention_debug.o: attention_debug.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@
attention_debug_hooks.o: attention_debug_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── kernel_summary ────────────────────────────────────────────────────────────
kernel_summary.so: kernel_summary.o kernel_summary_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 kernel_summary.o kernel_summary_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

kernel_summary.o: kernel_summary.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

kernel_summary_hooks.o: kernel_summary_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── attention_trace ───────────────────────────────────────────────────────────
attention_trace.so: attention_trace.o attention_trace_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 attention_trace.o attention_trace_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

attention_trace.o: attention_trace.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

attention_trace_hooks.o: attention_trace_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── sampling_mem_trace ────────────────────────────────────────────────────────
sampling_mem_trace.so: sampling_mem_trace.o sampling_mem_trace_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 sampling_mem_trace.o sampling_mem_trace_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

sampling_mem_trace.o: sampling_mem_trace.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

sampling_mem_trace_hooks.o: sampling_mem_trace_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── branch_divergence ─────────────────────────────────────────────────────────
branch_divergence.so: branch_divergence.o branch_divergence_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 branch_divergence.o branch_divergence_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

branch_divergence.o: branch_divergence.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

branch_divergence_hooks.o: branch_divergence_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@

# ── nvlink_trace ──────────────────────────────────────────────────────────────
nvlink_trace.so: nvlink_trace.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 nvlink_trace.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

nvlink_trace.o: nvlink_trace.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@
clean:
	rm -f *.so *.o
