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

ARCH?=all

# List of tools to build
TOOLS=instr_count.so mem_trace.so sm_profiler.so

all: $(TOOLS)

# Generic rule for building .so from .cu
%.so: %.cu $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -dc -c -std=c++17 $(INCLUDES) -Xptxas -cloning=no \
		-Xcompiler -Wall -arch=$(ARCH) -O3 -Xcompiler -fPIC $< -o $*.o
	$(NVCC) -arch=$(ARCH) -O3 $*.o $(LIBS) $(NVCC_PATH) \
		-lcuda -lcudart_static -shared -o $@

clean:
	rm -f *.so *.o

.PHONY: all clean
