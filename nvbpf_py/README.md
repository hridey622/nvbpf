# NV-BPF Python DSL

This is a Python-first generator for simple NV-BPF tools.

It is intentionally a restricted DSL, not a compiler for arbitrary Python.

The current DSL supports four authoring styles:

- declare one tool in Python
- declare explicit NV-BPF maps such as arrays and per-CPU arrays
- declare one or more counters
- declare ring-buffer event schemas
- write restricted custom device hook logic in Python
- declare host-only CUDA API tracing specs
- generate two higher-level host analyzers for GEMM wave fit and orchestration
- generate a host `.cu`, a `_hooks.cu`, and a `Makefile`
- build the resulting `.so`

## Example

```python
from nvbpf_py import tool, counter


@tool("atomic_count")
class AtomicCount:
    atomics = counter(
        opcodes=["ATOM", "RED"],
        description="Count atomic and reduction instructions",
    )
```

## Supported Counter Predicates

- `opcodes=[...]`
- `loads=True`
- `stores=True`
- `branches=True`

In this MVP, those predicates are combined with OR semantics for a given
counter.

## Event Schemas

```python
samples = event(
    fields={
        "addr": "u64",
        "sm_id": "u32",
        "warp_id": "u32",
        "is_load": "u8",
    },
    capacity=4096,
)
```

Supported field types:

- `u8`
- `u16`
- `u32`
- `u64`
- `i32`
- `i64`
- `bool`

## Explicit Maps

The DSL now supports explicit map declarations for tools that need a more
structured host/device state layout than simple counters.

```python
sm_cta_entries = percpu_array(type_name="u64", length=1)
active_sm_bitmap = array(type_name="u64", length=4)
```

In this first version, explicit maps are mainly used by the higher-level GEMM
host analyzers described below.

## Custom Device Hooks

Custom hook bodies are written in restricted Python and lowered into generated
CUDA device helpers.

Example:

```python
@device_hook(loads=True, stores=True)
def on_memory(pred, addr, is_load, sm_id, warp_id):
    if active_lanes == 0:
        return
    count("sampled")
    emit("samples", addr=addr, sm_id=sm_id, warp_id=warp_id, is_load=is_load)
```

Supported statements:

- `if` / `else`
- `return`
- simple assignments like `tmp = expr`
- `count("counter_name")`
- `emit("event_name", field=value, ...)`

Useful built-in variables inside hook bodies:

- `pred`
- `addr`
- `is_load`
- `active_lanes`
- `sm_id`
- `warp_id`
- `cta_id_x`
- `cta_id_y`
- `cta_id_z`

The hook language is intentionally small right now; it is meant to cover common
instrumentation patterns without asking users to write CUDA directly.

## Host-only CUDA API Traces

```python
peer_copy = api_trace(
    callbacks=["API_CUDA_cuMemcpyPeer", "API_CUDA_cuMemcpyPeerAsync"],
    correlate_launches=True,
)
```

This generates a host-only tool that watches CUDA API callbacks and can
correlate recent traced events with kernel launches.

## Higher-level GEMM Host Analyzers

The DSL can now generate two richer host-side tools directly:

### GEMM wave-fit trace

```python
from nvbpf_py import array, gemm_wavefit, percpu_array, tool


@tool("gemm_wavefit_trace_py", banner="GEMM_WAVEFIT_TRACE_PY")
class GemmWavefitTracePy:
    sm_cta_entries = percpu_array(type_name="u64", length=1)
    active_sm_bitmap = array(type_name="u64", length=4)
    analysis = gemm_wavefit(
        sm_cta_entries_map="sm_cta_entries",
        active_sm_bitmap_map="active_sm_bitmap",
    )
```

This generates a specialized tool that estimates resident-wave fit, reports
`fill_fraction`, and tracks how evenly CTAs are spread across SMs.
It defaults to one compact line per unique kernel shape; use `NVBPF_VERBOSE=1`
for the raw per-launch dump.

### GEMM orchestration map

```python
from nvbpf_py import gemm_orchestration_map, tool


@tool("gemm_orchestration_map_py", banner="GEMM_ORCHESTRATION_MAP_PY")
class GemmOrchestrationMapPy:
    analysis = gemm_orchestration_map()
```

This generates a host-only neighborhood trace that labels nearby kernels as
`attention`, `gemm`, `copy`, `reduction`, `elementwise`, and related classes.
It also defaults to grouped summaries, with `NVBPF_VERBOSE=1` restoring the
full per-neighborhood listing.

## Build A Tool

From the repo root:

```bash
python3 -m nvbpf_py.cli build tools/nvbpf_py_examples/atomic_count.py
```

That generates:

- `tools/nvbpf_generated/atomic_count/atomic_count.cu`
- `tools/nvbpf_generated/atomic_count/atomic_count_hooks.cu`
- `tools/nvbpf_generated/atomic_count/Makefile`

Then:

```bash
cd tools/nvbpf_generated/atomic_count
make clean && make atomic_count.so
```

Or do both in one step:

```bash
python3 -m nvbpf_py.cli build --compile tools/nvbpf_py_examples/atomic_count.py
```

To rebuild an existing generated tool directory, add `--force`:

```bash
python3 -m nvbpf_py.cli build --force --compile tools/nvbpf_py_examples/atomic_count.py
```

To generate directly into `tools/nvbpf_examples` and patch its shared
`Makefile` automatically:

```bash
python3 -m nvbpf_py.cli build \
  --integrate-examples \
  --force \
  tools/nvbpf_py_examples/atomic_count.py
```

You can compile the integrated example in the same step:

```bash
python3 -m nvbpf_py.cli build \
  --integrate-examples \
  --force \
  --compile \
  tools/nvbpf_py_examples/atomic_count.py
```

## Run The Tool

```bash
cd /path/to/repo
ACK_CTX_INIT_LIMITATION=1 \
LD_PRELOAD=$(pwd)/tools/nvbpf_generated/atomic_count/atomic_count.so \
./your_cuda_app
```

## Best Use Cases Right Now

- opcode counters
- load/store counters
- small per-kernel summaries based on simple counters

There are example specs in:

- `tools/nvbpf_py_examples/atomic_count.py`
- `tools/nvbpf_py_examples/memory_mix.py`
- `tools/nvbpf_py_examples/sampling_mem_events.py`
- `tools/nvbpf_py_examples/peer_copy_trace.py`
- `tools/nvbpf_py_examples/gemm_wavefit.py`
- `tools/nvbpf_py_examples/gemm_orchestration.py`

## Not In This DSL Yet

- arbitrary Python in hooks
- cross-hook shared local state
- generic Python-authored host callbacks and launch-neighborhood logic
- reusable Python-level helpers for CUDA API parameter struct decoding
- automatic `LD_PRELOAD` run wrapper
