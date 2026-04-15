# Writing A New NV-BPF Tool

This directory is the fastest place to learn the NV-BPF authoring pattern.

Most tools here follow the same structure:

1. A host-side tool file, for example `kernel_summary.cu`
2. A device hook file, for example `kernel_summary_hooks.cu`
3. A `Makefile` target that builds `tool.so`

The workflow is:

1. declare maps in the host file
2. inject a small device helper into matching instructions
3. reset maps before launch
4. read maps after launch
5. print a human-readable summary

## Tool Types

There are two common styles in this directory.

### 1. Hook + map tools

Use this when you want to count instructions, sample memory, or track per-SM state.

Examples:
- `instr_count.cu` + `instr_count_hooks.cu`
- `kernel_summary.cu` + `kernel_summary_hooks.cu`
- `sampling_mem_trace.cu` + `sampling_mem_trace_hooks.cu`
- `branch_divergence.cu` + `branch_divergence_hooks.cu`

These tools usually use:
- `BPF_ARRAY(...)`
- `BPF_PERCPU_ARRAY(...)`
- `BPF_RINGBUF(...)`

### 2. Host-only tools

Use this when you want to observe CUDA API events or launch metadata without injecting device code.

Example:
- `nvlink_trace.cu`

This style does not need a `_hooks.cu` file.

## The Smallest Useful Pattern

For most new tools, copy one of these first:

- copy `kernel_summary.*` if you want per-kernel counters
- copy `sampling_mem_trace.*` if you want memory events
- copy `branch_divergence.*` if you want branch-only logic
- copy `nvlink_trace.cu` if you want host-side API tracing

Then change only:

1. the map declarations
2. the opcode or instruction selection logic
3. the printed summary

## Step By Step

### 1. Add a new host file

Example skeleton:

```cpp
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <unordered_set>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

BPF_ARRAY(total_events, uint64_t, 1);

extern "C" __device__ __noinline__ void my_count(int pred, uint64_t pcounter);

static pthread_mutex_t launch_mutex;
static std::unordered_set<CUfunction> already_instrumented;

static void reset_state() {
    total_events.reset();
}
```

The host file is where you:
- declare maps
- decide which instructions to instrument
- enable instrumentation at kernel launch
- read and print results after kernel completion

### 2. Add a hook file

Example skeleton:

```cpp
#include <stdint.h>
#include "utils/utils.h"

extern "C" __device__ __noinline__ void my_count(int pred, uint64_t pcounter) {
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), pred);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    if (first_laneid == laneid && __popc(predicate_mask) > 0) {
        atomicAdd((unsigned long long*)pcounter, 1ULL);
    }
}
```

The hook should stay small. Good hooks usually:
- do one thing
- update one or two counters
- avoid complex logic unless really needed

### 3. Instrument only the instructions you care about

Typical pattern:

```cpp
for (auto* instr : instrs) {
    if (!matches(instr)) continue;
    nvbit_insert_call(instr, "my_count", IPOINT_BEFORE);
    nvbit_add_call_arg_guard_pred_val(instr);
    nvbit_add_call_arg_const_val64(instr, (uint64_t)&total_events.data[0]);
}
```

Common selectors:
- `instr->isLoad()`
- `instr->isStore()`
- `instr->getOpcodeShort()`
- `instr->getMemorySpace()`

### 4. Reset before launch, read after launch

Typical pattern:

```cpp
if (!is_exit) {
    instrument_function_if_needed(ctx, func);
    reset_state();
    nvbit_enable_instrumented(ctx, func, true);
} else {
    cudaDeviceSynchronize();
    printf("count=%lu\n", *total_events.lookup(0));
}
```

### 5. Add a Makefile target

Hook-based tool:

```make
my_tool.so: my_tool.o my_tool_hooks.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 my_tool.o my_tool_hooks.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

my_tool.o: my_tool.cu
	$(NVCC) $(HOST_FLAGS) $< -o $@

my_tool_hooks.o: my_tool_hooks.cu
	$(NVCC) $(HOOK_FLAGS) $< -o $@
```

Host-only tool:

```make
my_tool.so: my_tool.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 my_tool.o \
	        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@
```

## Real Example: Atomic Counter

If you want a tool that counts only atomic instructions, the changes are small:

1. copy `kernel_summary.cu` and `kernel_summary_hooks.cu`
2. keep only one map:
   `BPF_ARRAY(total_atomics, uint64_t, 1);`
3. change the selector to:

```cpp
static bool is_atomic_opcode(const char* opcode) {
    return strncmp(opcode, "ATOM", 4) == 0 || strncmp(opcode, "RED", 3) == 0;
}
```

4. print:

```cpp
printf("[NVBPF] %s atomics=%lu\n", func_name, *total_atomics.lookup(0));
```

That is usually enough to get a first working version.

## Recommended Defaults

If you want tools that are easy for other people to use, prefer:

- one summary line per kernel
- `NVBPF_KERNEL_FILTER` support
- sampling instead of tracing every event
- `ACK_CTX_INIT_LIMITATION=1`
- `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1` for map-heavy tools

## Build And Run

From this directory:

```bash
make clean && make kernel_summary.so
```

Then from the repo root:

```bash
ACK_CTX_INIT_LIMITATION=1 \
NVBPF_KERNEL_FILTER=sgemm \
LD_PRELOAD=$(pwd)/tools/nvbpf_examples/kernel_summary.so \
python3 test-apps/attention_pytorch/attention_pytorch.py --backend math
```

## Which Example Should I Copy?

- `instr_count`: simplest counting pattern
- `kernel_summary`: best general template
- `sampling_mem_trace`: best memory-event template
- `branch_divergence`: best branch-analysis template
- `attention_trace`: best workload-specific template
- `nvlink_trace`: best host-only trace template

## Rule Of Thumb

If the question is:

- "count something in device code" -> use maps + hooks
- "sample or stream device events" -> use maps + hooks + ring buffer
- "watch CUDA API behavior" -> use a host-only tool

Start from the closest existing example, not from scratch.
