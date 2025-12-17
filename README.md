# NV-BPF: eBPF-style Wrapper for NVBit

## Summary
Implemented a **header-only library** that standardizes NVBit GPU instrumentation into familiar **eBPF-like abstractions**:
- **Maps** (shared memory)
- **Hooks / Probes** (`SEC` macros)

---

## Architecture

 NVBit : 
 ↓ 
 NV-BPF Layer 
 ↓
 User Code

**Core Concepts**
- `SEC_KPROBE(name)`
- `BPF_ARRAY(...)`
- Hook Registry
- Map Templates
- Auto-Loader
- `nvbit_insert_call`
- Managed Memory

---

## Files Created

| File | Purpose |
|-----|--------|
| `nvbpf.h` | Main entry point (includes all components) |
| `nvbpf_types.h` | Context structures, enums, error codes |
| `nvbpf_helpers.h` | Helper functions (`bpf_get_context`, etc.) |
| `nvbpf_maps.h` | Map types (ARRAY, HASH, PERCPU, RINGBUF) |
| `nvbpf_hooks.h` | `SEC` macros and hook registry |
| `nvbpf_loader.h` | Auto-injection backend |

---

## Example Tools

| File | Description |
|-----|------------|
| `instr_count.cu` | Warp instruction counter |
| `mem_trace.cu` | Memory tracer with ring buffer |
| `sm_profiler.cu` | Per-SM profiling |

---

## API Quick Reference

### Maps

```c
// Fixed array with bounds checking
BPF_ARRAY(name, uint64_t, 1024);

// Key-value hash map
BPF_HASH(name, KeyType, ValueType, 1024);

// Per-SM private arrays (reduces contention)
BPF_PERCPU_ARRAY(name, uint64_t, 1);

// GPU → CPU ring buffer
BPF_RINGBUF(name, EventStruct, 16384);
```

### Hooks (SEC Macros)
```c
// Kernel entry/exit
SEC_KPROBE(my_probe) { ... }
SEC_KRETPROBE(my_exit_probe) { ... }

// Memory tracepoints
SEC_TRACEPOINT_MEM_LOAD(trace_loads) { ... }
SEC_TRACEPOINT_MEM_STORE(trace_stores) { ... }

// Instruction tracepoints
SEC_TRACEPOINT_INSTR(count_all) { ... }
SEC_TRACEPOINT_OPCODE(count_fma, "FFMA") { ... }
```

### Helpers
```c
NvBpfContext ctx;
bpf_get_context(&ctx);            // Fill context
bpf_get_current_sm_id();          // SM ID
bpf_get_current_warp_id();        // Warp ID
bpf_ktime_get_ns();               // GPU timestamp
bpf_probe_read_kernel(&dst, src); // Safe memory read
bpf_printk("SM %d", ctx.sm_id);   // Debug output
```






