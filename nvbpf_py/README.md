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
@hook(loads=True, stores=True)
def on_memory(ctx):
    if ctx.active_lanes == 0:
        return
    count("sampled")
    ctx.atomic_add("site_totals", 0, ctx.active_lanes)
    ctx.emit(
        "samples",
        addr=ctx.addr,
        sm_id=ctx.sm_id,
        warp_id=ctx.warp_id,
        is_load=ctx.is_load,
    )
```

Supported statements:

- `if` / `else`
- `for i in range(...)`
- `break` / `continue`
- `return`
- `pass`
- simple assignments like `tmp = expr`
- augmented assignments like `tmp += 1`
- `count("counter_name")`
- `atomic_add("counter_name", value)`
- `atomic_add("map_name", index, value)`
- `map_set("map_name", index, value)`
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

`hook(...)` is the preferred alias; `device_hook(...)` still works.

Supported `ctx.*` hook API:

- attributes: `ctx.pred`, `ctx.addr`, `ctx.is_load`, `ctx.active_mask`,
  `ctx.predicate_mask`, `ctx.lane_id`, `ctx.active_lanes`, `ctx.sm_id`,
  `ctx.warp_id`, `ctx.cta_id_x/y/z`, `ctx.grid_dim_x/y/z`,
  `ctx.block_dim_x/y/z`
- statements: `ctx.count(...)`, `ctx.atomic_add(...)`, `ctx.emit(...)`
- statements: `ctx.map_set(...)`
- expressions: `ctx.ballot(expr)`, `ctx.popc(expr)`, `ctx.ffs(expr)`,
  `ctx.map_get(...)`

The older free-variable style like `active_lanes` and top-level `count(...)`
still works, but `ctx.*` is the preferred interface going forward.

Example with loops and map access:

```python
from nvbpf_py import array, counter, hook, map_value, on_launch_exit, short_kernel_name, tool


@tool("loop_bucket_report_py", banner="LOOP_BUCKET_REPORT_PY")
class LoopBucketReportPy:
    sampled = counter(loads=True)
    lane_bucket_hits = array(type_name="u64", length=4)
    seen_mask = array(type_name="u64", length=1)

    @hook(loads=True)
    def on_load(ctx):
        for bucket in range(4):
            lower = bucket * 8
            upper = (bucket + 1) * 8
            if ctx.active_lanes > lower and ctx.active_lanes <= upper:
                ctx.atomic_add("lane_bucket_hits", bucket, 1)
                current = ctx.map_get("seen_mask", 0)
                ctx.map_set("seen_mask", 0, current | (1 << bucket))

    @on_launch_exit()
    def report():
        print(
            "kernel=", short_kernel_name(),
            "b0=", map_value("lane_bucket_hits", 0),
            "b1=", map_value("lane_bucket_hits", 1),
            "b2=", map_value("lane_bucket_hits", 2),
            "b3=", map_value("lane_bucket_hits", 3),
            "seen_mask=", map_value("seen_mask", 0),
        )
```

## Launch-exit Reporting

You can now attach a host-side launch-exit callback to a normal Python-authored
tool and print custom summaries without hand-writing the `.cu` host file.

Example:

```python
from nvbpf_py import (
    array,
    counter,
    counter_value,
    grid_dim_x,
    hook,
    map_value,
    on_launch_enter,
    on_launch_exit,
    short_kernel_name,
    tool,
)


@tool("python_only_report", banner="PYTHON_ONLY_REPORT")
class PythonOnlyReport:
    sampled = counter(loads=True)
    site_totals = array(type_name="u64", length=1)

    @hook(loads=True)
    def on_load(ctx):
        if ctx.active_lanes == 0:
            return
        ctx.atomic_add("site_totals", 0, ctx.active_lanes)

    @on_launch_enter()
    def announce():
        print("launch", short_kernel_name(), "grid_x=", grid_dim_x())

    @on_launch_exit()
    def report():
        print(
            "kernel=", short_kernel_name(),
            "grid_x=", grid_dim_x(),
            "sampled=", counter_value("sampled"),
            "lanes=", map_value("site_totals", 0),
        )
```

Supported launch-enter/launch-exit statements:

- `if` / `else`
- `return`
- simple assignments like `tmp = expr`
- `print(...)`

Useful launch-exit helpers:

- `counter_value("counter_name")`
- `map_value("map_name", index)`
- `kernel_name()`
- `short_kernel_name()`
- `grid_dim_x()` / `grid_dim_y()` / `grid_dim_z()`
- `block_dim_x()` / `block_dim_y()` / `block_dim_z()`
- `regs()`
- `smem_static()`
- `smem_dynamic()`

`@on_launch_enter()` and `@on_launch_exit()` each support one callback per tool
in this version.

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

### Epilogue fusion trace

```python
from nvbpf_py import epilogue_fusion_trace, tool


@tool("epilogue_fusion_trace_py", banner="EPILOGUE_FUSION_TRACE_PY")
class EpilogueFusionTracePy:
    analysis = epilogue_fusion_trace()
```

This generates a host-only fusion-focused summary that distinguishes
`fused_likely` from separate post-kernel bias/activation/scale/copy paths.
Use `NVBPF_EPILOGUE_WINDOW=3` to widen or narrow the post-kernel window.

### Tail-fragment tracker

```python
from nvbpf_py import tail_fragment_tracker, tool


@tool("tail_fragment_tracker_py", banner="TAIL_FRAGMENT_TRACKER_PY")
class TailFragmentTrackerPy:
    analysis = tail_fragment_tracker()
```

This generates the same grouped tail/edge inefficiency summary as the
handwritten tool. Use `NVBPF_TAIL_ACTIVE_LANES=16` to adjust the "low active
lanes" threshold and `NVBPF_VERBOSE=1` for the per-launch detail dump.

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
python3 -m nvbpf_py.cli run \
  tools/nvbpf_py_examples/atomic_count.py \
  -- ./test-apps/vectoradd/matrix_add
```

This rebuilds the generated tool, compiles it, sets `LD_PRELOAD` for you, and
runs the target command from the repo root. If you already built the `.so`, you
can reuse it:

```bash
python3 -m nvbpf_py.cli run \
  --no-build \
  tools/nvbpf_py_examples/atomic_count.py \
  -- ./test-apps/vectoradd/matrix_add
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
- `tools/nvbpf_py_examples/epilogue_fusion.py`
- `tools/nvbpf_py_examples/tail_fragment.py`
- `tools/nvbpf_py_examples/python_only_report.py`
- `tools/nvbpf_py_examples/loop_bucket_report.py`

## Not In This DSL Yet

- arbitrary Python in hooks
- cross-hook shared local state
- generic Python-authored host callbacks and launch-neighborhood logic
- reusable Python-level helpers for CUDA API parameter struct decoding
- arbitrary Python-authored launch-neighborhood analyzers
