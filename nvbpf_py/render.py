from __future__ import annotations

import keyword
import os
from pathlib import Path
import re

from .model import (
    ApiTraceSpec,
    CounterSpec,
    DeviceHookSpec,
    EventFieldSpec,
    EventSpec,
    MapSpec,
    ToolSpec,
)
from .transpile import HookRender, render_custom_hook


_TYPE_MAP = {
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "bool": "uint8_t",
}

_PRINTF_MAP = {
    "u8": "%u",
    "u16": "%u",
    "u32": "%u",
    "u64": "%lu",
    "i32": "%d",
    "i64": "%ld",
    "bool": "%u",
}


def _sanitize_ident(name: str) -> str:
    ident = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    if not ident:
        ident = "tool"
    if ident[0].isdigit():
        ident = f"tool_{ident}"
    if keyword.iskeyword(ident):
        ident = f"{ident}_tool"
    return ident


def _quote_cpp(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _has_device_part(tool: ToolSpec) -> bool:
    return bool(tool.counters or tool.events or tool.device_hooks or tool.gemm_wavefit)


def _has_hook_file(tool: ToolSpec) -> bool:
    return _has_device_part(tool)


def _map_decl(map_spec: MapSpec) -> str:
    macro = {
        "array": "BPF_ARRAY",
        "percpu_array": "BPF_PERCPU_ARRAY",
    }.get(map_spec.kind)
    if macro is None:
        raise RuntimeError(f"unsupported map kind: {map_spec.kind}")
    type_name = _TYPE_MAP.get(map_spec.type_name)
    if type_name is None:
        raise RuntimeError(
            f"unsupported map type {map_spec.type_name!r} for map {map_spec.name!r}"
        )
    suffix = f"  // {map_spec.description}" if map_spec.description else ""
    return f"{macro}({map_spec.name}, {type_name}, {map_spec.length});{suffix}"


def _map_by_name(tool: ToolSpec, name: str) -> MapSpec:
    for map_spec in tool.maps:
        if map_spec.name == name:
            return map_spec
    raise RuntimeError(f"tool {tool.name!r} references unknown map {name!r}")


def _event_type(event: EventSpec) -> str:
    return f"{event.name}_event_t"


def _event_structs(tool: ToolSpec) -> str:
    blocks: list[str] = []
    for event in tool.events:
        fields = "\n".join(
            f"    {_TYPE_MAP[field.type_name]} {field.name};"
            for field in event.fields
        )
        blocks.append(
            "\n".join(
                [
                    f"struct {_event_type(event)} {{",
                    fields,
                    "};",
                ]
            )
        )
    return "\n\n".join(blocks)


def _counter_comment(counter: CounterSpec) -> str:
    pieces: list[str] = []
    if counter.opcodes:
        pieces.append("opcodes=" + ",".join(counter.opcodes))
    if counter.loads:
        pieces.append("loads")
    if counter.stores:
        pieces.append("stores")
    if counter.branches:
        pieces.append("branches")
    if counter.description:
        pieces.append(counter.description)
    return " | ".join(pieces)


def _counter_match_expr(counter: CounterSpec) -> str:
    checks: list[str] = []
    if counter.opcodes:
        opcode_checks = " || ".join(
            f'opcode_starts_with(opcode, "{_quote_cpp(prefix)}")'
            for prefix in counter.opcodes
        )
        checks.append(f"({opcode_checks})")
    if counter.loads:
        load_expr = "instr->isLoad()"
        if counter.exclude_constant_loads:
            load_expr += " && instr->getMemorySpace() != InstrType::MemorySpace::CONSTANT"
        checks.append(f"({load_expr})")
    if counter.stores:
        checks.append("(instr->isStore())")
    if counter.branches:
        checks.append("(is_branch_opcode(opcode))")
    return " || ".join(checks) if checks else "false"


def _hook_match_expr(hook: DeviceHookSpec) -> str:
    checks: list[str] = []
    if hook.opcodes:
        opcode_checks = " || ".join(
            f'opcode_starts_with(opcode, "{_quote_cpp(prefix)}")'
            for prefix in hook.opcodes
        )
        checks.append(f"({opcode_checks})")
    if hook.loads:
        load_expr = "instr->isLoad()"
        if hook.exclude_constant_loads:
            load_expr += " && instr->getMemorySpace() != InstrType::MemorySpace::CONSTANT"
        checks.append(f"({load_expr})")
    if hook.stores:
        checks.append("(instr->isStore())")
    if hook.branches:
        checks.append("(is_branch_opcode(opcode))")
    return " || ".join(checks) if checks else "false"


def _render_counter_matchers(tool: ToolSpec) -> str:
    blocks: list[str] = []
    for counter in tool.counters:
        blocks.append(
            "\n".join(
                [
                    f"static bool matches_{counter.name}(Instr* instr) {{",
                    '    const char* opcode = instr->getOpcodeShort();',
                    f"    return {_counter_match_expr(counter)};",
                    "}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _render_hook_matchers(tool: ToolSpec) -> str:
    blocks: list[str] = []
    for hook in tool.device_hooks:
        blocks.append(
            "\n".join(
                [
                    f"static bool matches_hook_{hook.name}(Instr* instr) {{",
                    '    const char* opcode = instr->getOpcodeShort();',
                    f"    return {_hook_match_expr(hook)};",
                    "}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _render_printf_value(field: EventFieldSpec) -> str:
    if field.type_name == "bool":
        return f"(unsigned)evt->{field.name}"
    if field.type_name in ("u8", "u16", "u32"):
        return f"(unsigned)evt->{field.name}"
    if field.type_name == "u64":
        return f"(unsigned long)evt->{field.name}"
    if field.type_name == "i64":
        return f"(long)evt->{field.name}"
    return f"evt->{field.name}"


def _render_event_consume(event: EventSpec) -> str:
    event_ty = _event_type(event)
    field_fmt = " ".join(
        f"{field.name}={_PRINTF_MAP[field.type_name]}" for field in event.fields
    )
    field_vals = ", ".join(_render_printf_value(field) for field in event.fields)
    verbose_printer = (
        f'printf("          {event.name} {field_fmt}\\n", {field_vals});'
        if field_vals
        else f'printf("          {event.name}\\n");'
    )
    return f"""
        uint64_t {event.name}_count = 0;
        if (verbose) {{
            {event.name}_count = {event.name}.consume([]({event_ty}* evt) {{
                {verbose_printer}
            }});
        }} else {{
            {event.name}_count = {event.name}.consume([]({event_ty}*) {{}});
        }}
        printf("        {event.name}_events=%lu dropped=%lu\\n",
               {event.name}_count, {event.name}.dropped);
"""


def _render_api_trace_statics(tool: ToolSpec) -> str:
    if not tool.api_traces:
        return ""
    blocks = [
        "struct RecentApiTrace {",
        "    uint64_t event_id = 0;",
        "    bool valid = false;",
        "};",
        "static uint64_t api_event_counter = 0;",
    ]
    for trace in tool.api_traces:
        blocks.append(f"static uint64_t {trace.name}_hits = 0;")
        if trace.correlate_launches:
            blocks.append(f"static RecentApiTrace recent_{trace.name};")
    return "\n".join(blocks)


def _render_api_trace_body(tool: ToolSpec) -> str:
    if not tool.api_traces:
        return ""
    lines = [
        "    if (is_exit) api_event_counter++;",
    ]
    for trace in tool.api_traces:
        cond = " || ".join(f"cbid == {cb}" for cb in trace.callbacks)
        lines.extend(
            [
                f"    if (is_exit == {'1' if trace.on_exit else '0'} && ({cond})) {{",
                f"        {trace.name}_hits++;",
            ]
        )
        if trace.correlate_launches:
            lines.extend(
                [
                    f"        recent_{trace.name}.event_id = api_event_counter;",
                    f"        recent_{trace.name}.valid = true;",
                ]
            )
        lines.extend(
            [
                f'        printf("[NVBPF] api_trace {trace.name} event=%s\\n", name);',
                "    }",
            ]
        )
    lines.append("    if (is_launch && is_exit) {")
    lines.append("        CUfunction launch_func = nvbpf_get_launch_func(cbid, params);")
    lines.append('        const char* launch_name = nvbit_get_func_name(ctx, launch_func);')
    lines.append("        if (kernel_name_filter.empty() || strstr(launch_name, kernel_name_filter.c_str()) != nullptr) {")
    for trace in tool.api_traces:
        if trace.correlate_launches:
            lines.extend(
                [
                    f"            if (recent_{trace.name}.valid && api_event_counter - recent_{trace.name}.event_id <= 8) {{",
                    f'                printf("        correlated_{trace.name}=1 delta_events=%lu kernel=%s\\n",',
                    f"                       api_event_counter - recent_{trace.name}.event_id, launch_name);",
                    "            }",
                ]
            )
    lines.append("        }")
    lines.append("    }")
    return "\n".join(lines)


def _render_api_trace_term(tool: ToolSpec) -> str:
    if not tool.api_traces:
        return ""
    return "\n".join(
        f'    printf("[NVBPF {tool.banner}] api_trace {trace.name} hits=%lu\\n", {trace.name}_hits);'
        for trace in tool.api_traces
    )


def _render_hook_call(
    prefix: str,
    hook: DeviceHookSpec,
    hook_render: HookRender,
    call_is_load: int | None,
    indent: str = "                ",
) -> list[str]:
    lines = [
        f'{indent}nvbit_insert_call(instr, "{prefix}_{hook.name}", IPOINT_BEFORE);',
        f"{indent}nvbit_add_call_arg_guard_pred_val(instr);",
    ]
    if hook_render.needs_addr:
        lines.append(f"{indent}nvbit_add_call_arg_mref_addr64(instr, 0);")
    if hook_render.needs_is_load:
        if call_is_load is None:
            raise RuntimeError(f"hook {hook.name} uses is_load but is not a memory hook")
        lines.append(f"{indent}nvbit_add_call_arg_const_val32(instr, {call_is_load});")
    return lines


def _render_map_arg_injection(
    tool: ToolSpec,
    indent: str = "                ",
) -> list[str]:
    lines: list[str] = []
    for counter in tool.counters:
        lines.append(
            f"{indent}nvbit_add_call_arg_const_val64(instr, (uint64_t)&{counter.name}.data[0]);"
        )
    for event in tool.events:
        lines.append(
            f"{indent}nvbit_add_call_arg_const_val64(instr, (uint64_t)&{event.name});"
        )
    return lines


def _render_device_instrumentation(tool: ToolSpec, prefix: str) -> str:
    lines: list[str] = []

    for counter in tool.counters:
        lines.extend(
            [
                f"            if (matches_{counter.name}(instr)) {{",
                f'                nvbit_insert_call(instr, "{prefix}_count_counter", IPOINT_BEFORE);',
                "                nvbit_add_call_arg_guard_pred_val(instr);",
                f"                nvbit_add_call_arg_const_val64(instr, (uint64_t)&{counter.name}.data[0]);",
                "            }",
            ]
        )

    events_by_name = {event.name: event for event in tool.events}
    counters_by_name = {counter.name for counter in tool.counters}
    for hook in tool.device_hooks:
        hook_render = render_custom_hook(hook, events_by_name, counters_by_name)
        if hook.loads or hook.stores:
            if hook.loads:
                lines.append(
                    "            if (instr->isLoad()"
                    + (" && instr->getMemorySpace() != InstrType::MemorySpace::CONSTANT" if hook.exclude_constant_loads else "")
                    + f" && matches_hook_{hook.name}(instr)) {{"
                )
                lines.extend(_render_hook_call(prefix, hook, hook_render, 1))
                lines.extend(_render_map_arg_injection(tool))
                lines.append("            }")
            if hook.stores:
                lines.append(f"            if (instr->isStore() && matches_hook_{hook.name}(instr)) {{")
                lines.extend(_render_hook_call(prefix, hook, hook_render, 0))
                lines.extend(_render_map_arg_injection(tool))
                lines.append("            }")
        else:
            lines.append(f"            if (matches_hook_{hook.name}(instr)) {{")
            lines.extend(_render_hook_call(prefix, hook, hook_render, None))
            lines.extend(_render_map_arg_injection(tool))
            lines.append("            }")

    return "\n".join(lines)


def _render_gemm_wavefit_host(tool: ToolSpec) -> str:
    assert tool.gemm_wavefit is not None
    analysis = tool.gemm_wavefit
    prefix = _sanitize_ident(tool.name)
    sm_entries = _map_by_name(tool, analysis.sm_cta_entries_map)
    bitmap = _map_by_name(tool, analysis.active_sm_bitmap_map)
    map_decls = "\n".join(_map_decl(map_spec) for map_spec in tool.maps)
    bitmap_words = bitmap.length

    return f"""/*
 * Auto-generated by nvbpf_py. Edit the Python spec, not this file.
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

{map_decls}

extern "C" __device__ __noinline__ void {prefix}_kernel_entry(int pred,
                                                              uint64_t psm_entries,
                                                              uint64_t pbitmap);

static pthread_mutex_t launch_mutex;
static std::unordered_set<CUfunction> already_instrumented;
static std::string filter_csv =
    "{_quote_cpp(analysis.filter_csv)}";
static bool verbose = false;
static bool full_names = false;
static uint64_t matched_launches = 0;

struct WavefitSummary {{
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
}};

static std::vector<WavefitSummary> summaries;

static bool csv_match(const char* name, const std::string& csv) {{
    size_t start = 0;
    while (start < csv.size()) {{
        size_t end = csv.find(',', start);
        if (end == std::string::npos) end = csv.size();
        std::string tok = csv.substr(start, end - start);
        if (!tok.empty() && strstr(name, tok.c_str()) != nullptr) return true;
        start = end + 1;
    }}
    return false;
}}

static std::string compact_kernel_name(const std::string& raw) {{
    if (full_names) return raw;
    std::string name = raw;
    if (name.rfind("void ", 0) == 0) {{
        name = name.substr(5);
    }}
    size_t paren = name.find('(');
    if (paren != std::string::npos) {{
        name = name.substr(0, paren);
    }}
    if (name.size() <= 56) return name;
    return name.substr(0, 24) + "..." + name.substr(name.size() - 24);
}}

static const char* heuristic_label(int code) {{
    switch (code) {{
        case 0: return "underfill";
        case 1: return "perfect_fit";
        case 2: return "small_tail";
        default: return "large_tail";
    }}
}}

static WavefitSummary* find_summary(const char* func_name,
                                    int gx, int gy, int gz,
                                    int bx, int by, int bz,
                                    uint32_t regs,
                                    uint32_t smem_static,
                                    uint32_t smem_dynamic) {{
    for (auto& summary : summaries) {{
        if (summary.kernel_name == func_name &&
            summary.gx == gx && summary.gy == gy && summary.gz == gz &&
            summary.bx == bx && summary.by == by && summary.bz == bz &&
            summary.regs == regs &&
            summary.smem_static == smem_static &&
            summary.smem_dynamic == smem_dynamic) {{
            return &summary;
        }}
    }}
    return nullptr;
}}

static void reset_state() {{
    {sm_entries.name}.reset();
    {bitmap.name}.reset();
}}

static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {{
    if (!already_instrumented.insert(func).second) {{
        return;
    }}

    const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, func);
    if (instrs.empty()) {{
        return;
    }}

    nvbit_insert_call(instrs[0], "{prefix}_kernel_entry", IPOINT_BEFORE);
    nvbit_add_call_arg_guard_pred_val(instrs[0]);
    nvbit_add_call_arg_const_val64(instrs[0], (uint64_t)&{sm_entries.name}.data[0][0]);
    nvbit_add_call_arg_const_val64(instrs[0], (uint64_t)&{bitmap.name}.data[0]);
}}

static void launch_dims(nvbit_api_cuda_t cbid, void* params,
                        int* gx, int* gy, int* gz,
                        int* bx, int* by, int* bz) {{
    *gx = *gy = *gz = *bx = *by = *bz = 1;
    if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {{
        cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
        *gx = p->config->gridDimX;
        *gy = p->config->gridDimY;
        *gz = p->config->gridDimZ;
        *bx = p->config->blockDimX;
        *by = p->config->blockDimY;
        *bz = p->config->blockDimZ;
    }} else if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel) {{
        cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
        *gx = p->gridDimX;
        *gy = p->gridDimY;
        *gz = p->gridDimZ;
        *bx = p->blockDimX;
        *by = p->blockDimY;
        *bz = p->blockDimZ;
    }}
}}

void nvbit_at_init() {{
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    pthread_mutex_init(&launch_mutex, nullptr);
    if (const char* env = getenv("{_quote_cpp(analysis.filter_env)}")) {{
        filter_csv = env;
    }}
    verbose = getenv("NVBPF_VERBOSE") != nullptr;
    full_names = getenv("NVBPF_FULL_NAMES") != nullptr;
    printf("[NVBPF {tool.banner}] Tool loaded\\n");
}}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {{
    if (!nvbpf_is_launch_event(cbid)) return;
    CUfunction func = nvbpf_get_launch_func(cbid, params);
    const char* func_name = nvbit_get_func_name(ctx, func);

    if (!is_exit) {{
        bool match = csv_match(func_name, filter_csv);
        pthread_mutex_lock(&launch_mutex);
        if (match) {{
            instrument_function_if_needed(ctx, func);
            reset_state();
            nvbit_enable_instrumented(ctx, func, true);
        }} else {{
            nvbit_enable_instrumented(ctx, func, false);
        }}
        if (!match) {{
            pthread_mutex_unlock(&launch_mutex);
        }}
    }} else {{
        cudaDeviceSynchronize();
        if (!csv_match(func_name, filter_csv)) {{
            return;
        }}
        matched_launches++;

        int gx, gy, gz, bx, by, bz;
        launch_dims(cbid, params, &gx, &gy, &gz, &bx, &by, &bz);
        uint64_t total_ctas = (uint64_t)gx * gy * gz;

        CUdevice dev = 0;
        cuCtxGetDevice(&dev);

        int sm_count = 0;
        cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);

        func_config_t cfg{{}};
        nvbit_get_func_config(ctx, func, &cfg);

        int resident_ctas_per_sm = 1;
        int threads_per_block = bx * by * bz;
        if (threads_per_block > 0) {{
            CUresult occ_status = cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &resident_ctas_per_sm, func, threads_per_block, cfg.shmem_dynamic_nbytes);
            if (occ_status != CUDA_SUCCESS || resident_ctas_per_sm <= 0) {{
                resident_ctas_per_sm = 1;
            }}
        }}

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
        for (int word = 0; word < {bitmap_words}; word++) {{
            uint64_t* bm = {bitmap.name}.lookup(word);
            if (bm) active_sms += __builtin_popcountll(*bm);
        }}

        uint64_t sum_entries = 0;
        uint64_t min_entries = UINT64_MAX;
        uint64_t max_entries = 0;
        int used_sms = 0;
        for (int sm = 0; sm < sm_count; sm++) {{
            uint64_t* entries = {sm_entries.name}.lookup_sm(sm, 0);
            uint64_t value = entries ? *entries : 0;
            if (value == 0) continue;
            used_sms++;
            sum_entries += value;
            if (value < min_entries) min_entries = value;
            if (value > max_entries) max_entries = value;
        }}
        if (min_entries == UINT64_MAX) min_entries = 0;

        double avg_entries =
            used_sms > 0 ? (double)sum_entries / (double)used_sms : 0.0;

        int heuristic_code = 0;
        if (full_waves == 0 && tail_ctas > 0) {{
            heuristic_code = 0;
        }} else if (tail_ctas == 0) {{
            heuristic_code = 1;
        }} else if (tail_fraction < 0.25) {{
            heuristic_code = 2;
        }} else {{
            heuristic_code = 3;
        }}

        if (verbose) {{
            printf("[NVBPF] gemm_wavefit kernel=%s\\n", func_name);
            printf("        launch: grid=(%d,%d,%d) block=(%d,%d,%d) total_ctas=%lu regs=%u smem=%u+%u\\n",
                   gx, gy, gz, bx, by, bz, total_ctas, cfg.num_registers,
                   cfg.shmem_static_nbytes, cfg.shmem_dynamic_nbytes);
            printf("        waves: sms=%d resident_ctas_per_sm=%d wave_capacity=%lu full_waves=%lu tail_ctas=%lu tail_fraction=%.3f fill_fraction=%.3f\\n",
                   sm_count, resident_ctas_per_sm, wave_capacity, full_waves,
                   tail_ctas, tail_fraction, launch_fill_fraction);
            printf("        distribution: active_sms=%d used_sms=%d cta_entries=%lu min=%lu avg=%.2f max=%lu tail_empty_slots=%lu\\n",
                   active_sms, used_sms, sum_entries, min_entries, avg_entries, max_entries,
                   tail_empty_slots);
            if (heuristic_code == 0) {{
                printf("        heuristic: launch does not fill a single resident wave; severe wave underfill dominates utilization\\n");
            }} else if (heuristic_code == 1) {{
                printf("        heuristic: perfect wave fit for estimated resident CTA capacity\\n");
            }} else if (heuristic_code == 2) {{
                printf("        heuristic: small tail wave; underfill is limited\\n");
            }} else {{
                printf("        heuristic: sizable partial tail wave; launch shape likely wastes a noticeable final wave\\n");
            }}
        }} else {{
            WavefitSummary* summary = find_summary(
                func_name, gx, gy, gz, bx, by, bz, cfg.num_registers,
                cfg.shmem_static_nbytes, cfg.shmem_dynamic_nbytes);
            if (summary == nullptr) {{
                WavefitSummary fresh{{}};
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
            }}
            summary->launches++;
        }}
        if (used_sms > 0 && max_entries > min_entries + 1) {{
            if (verbose) {{
                printf("        heuristic: CTA distribution is uneven across active SMs\\n");
            }}
        }}
        pthread_mutex_unlock(&launch_mutex);
    }}
}}

void nvbit_at_term() {{
    if (!verbose) {{
        printf("[NVBPF {tool.banner}] matched_launches=%lu unique_kernels=%zu\\n",
               matched_launches, summaries.size());
        for (const auto& summary : summaries) {{
            printf("  x%-3lu %-32s | ctas=%-4lu fill=%.3f sms=%d/%d regs=%u smem=%u+%u | %s\\n",
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
        }}
    }}
    printf("[NVBPF {tool.banner}] Tool terminated\\n");
}}
"""


def _render_gemm_wavefit_hook(tool: ToolSpec) -> str:
    assert tool.gemm_wavefit is not None
    prefix = _sanitize_ident(tool.name)
    bitmap = _map_by_name(tool, tool.gemm_wavefit.active_sm_bitmap_map)
    bitmap_words = bitmap.length
    max_sms = bitmap_words * 64
    return f"""/*
 * Auto-generated by nvbpf_py. Edit the Python spec, not this file.
 */

#include <stdint.h>
#include "nvbpf_helpers.h"

static constexpr uint32_t kNvbpfMaxSms = {max_sms};
static constexpr uint32_t kBitmapWords = {bitmap_words};

extern "C" __device__ __noinline__ void {prefix}_kernel_entry(int pred,
                                                              uint64_t psm_entries,
                                                              uint64_t pbitmap) {{
    if (!pred) return;
    if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) return;

    uint32_t sm = bpf_get_current_sm_id();
    if (sm >= kNvbpfMaxSms) return;

    uint64_t* sm_entries = (uint64_t*)psm_entries;
    atomicAdd((unsigned long long*)&sm_entries[sm], 1ULL);

    uint32_t word = sm / 64;
    if (word < kBitmapWords) {{
        uint64_t* bitmap = (uint64_t*)pbitmap;
        atomicOr((unsigned long long*)&bitmap[word], 1ULL << (sm % 64));
    }}
}}
"""


def _render_gemm_orchestration_host(tool: ToolSpec) -> str:
    assert tool.gemm_orchestration is not None
    analysis = tool.gemm_orchestration
    return f"""/*
 * Auto-generated by nvbpf_py. Edit the Python spec, not this file.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

enum LaunchClass {{
    LC_ATTENTION = 0,
    LC_GEMM = 1,
    LC_EPILOGUE = 2,
    LC_COPY = 3,
    LC_TRANSPOSE = 4,
    LC_REDUCTION = 5,
    LC_ELEMENTWISE = 6,
    LC_OTHER = 7,
}};

struct LaunchRecord {{
    uint64_t event_id = 0;
    int gpu = -1;
    LaunchClass klass = LC_OTHER;
    bool is_kernel = false;
    size_t bytes = 0;
    std::string name;
}};

static std::string filter_csv =
    "{_quote_cpp(analysis.filter_csv)}";
static int neighborhood_window = {analysis.default_window};
static uint64_t api_event_counter = 0;
static std::vector<LaunchRecord> records;
static bool verbose = false;
static bool full_names = false;

struct NeighborhoodAggregate {{
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
}};

static std::vector<NeighborhoodAggregate> aggregates;

static bool csv_match(const char* name, const std::string& csv) {{
    size_t start = 0;
    while (start < csv.size()) {{
        size_t end = csv.find(',', start);
        if (end == std::string::npos) end = csv.size();
        std::string tok = csv.substr(start, end - start);
        if (!tok.empty() && strstr(name, tok.c_str()) != nullptr) return true;
        start = end + 1;
    }}
    return false;
}}

static std::string compact_kernel_name(const std::string& raw) {{
    if (full_names) return raw;
    std::string name = raw;
    if (name.rfind("void ", 0) == 0) {{
        name = name.substr(5);
    }}
    size_t paren = name.find('(');
    if (paren != std::string::npos) {{
        name = name.substr(0, paren);
    }}
    if (name.size() <= 56) return name;
    return name.substr(0, 24) + "..." + name.substr(name.size() - 24);
}}

static void update_range(int value, int* min_value, int* max_value) {{
    if (value < *min_value) *min_value = value;
    if (value > *max_value) *max_value = value;
}}

static void format_range(char* out, size_t out_size, int min_value, int max_value) {{
    if (min_value == max_value) {{
        snprintf(out, out_size, "%d", min_value);
    }} else {{
        snprintf(out, out_size, "%d-%d", min_value, max_value);
    }}
}}

static NeighborhoodAggregate* find_aggregate(const LaunchRecord& rec) {{
    for (auto& agg : aggregates) {{
        if (agg.center_class == rec.klass && agg.center_name == rec.name) {{
            return &agg;
        }}
    }}
    return nullptr;
}}

static LaunchClass classify_name(const char* name) {{
    if (strstr(name, "fmha") || strstr(name, "flash") || strstr(name, "attention") ||
        strstr(name, "attn")) {{
        return LC_ATTENTION;
    }}
    if (strstr(name, "gemm") || strstr(name, "sgemm") || strstr(name, "matmul") ||
        strstr(name, "cublas") || strstr(name, "cutlass") || strstr(name, "wmma")) {{
        return LC_GEMM;
    }}
    if (strstr(name, "epilogue") || strstr(name, "bias") || strstr(name, "relu") ||
        strstr(name, "gelu") || strstr(name, "silu") || strstr(name, "clamp") ||
        strstr(name, "activation")) {{
        return LC_EPILOGUE;
    }}
    if (strstr(name, "copy") || strstr(name, "cast") || strstr(name, "memcpy") ||
        strstr(name, "reformat") || strstr(name, "convert")) {{
        return LC_COPY;
    }}
    if (strstr(name, "transpose") || strstr(name, "permute") ||
        strstr(name, "layout")) {{
        return LC_TRANSPOSE;
    }}
    if (strstr(name, "reduce") || strstr(name, "reduction") ||
        strstr(name, "softmax") || strstr(name, "layernorm")) {{
        return LC_REDUCTION;
    }}
    if (strstr(name, "elementwise") || strstr(name, "vectorized") ||
        strstr(name, "unrolled_elementwise") || strstr(name, "binary") ||
        strstr(name, "unary") || strstr(name, "mul") || strstr(name, "add")) {{
        return LC_ELEMENTWISE;
    }}
    return LC_OTHER;
}}

static const char* class_name(LaunchClass klass) {{
    switch (klass) {{
        case LC_ATTENTION: return "attention";
        case LC_GEMM: return "gemm";
        case LC_EPILOGUE: return "epilogue";
        case LC_COPY: return "copy";
        case LC_TRANSPOSE: return "transpose";
        case LC_REDUCTION: return "reduction";
        case LC_ELEMENTWISE: return "elementwise";
        default: return "other";
    }}
}}

static void record_api_copy(const char* api_name, size_t bytes) {{
    LaunchRecord rec{{}};
    rec.event_id = api_event_counter;
    rec.klass = LC_COPY;
    rec.bytes = bytes;
    rec.name = api_name;
    records.push_back(rec);
}}

static bool is_focus_gemm(const LaunchRecord& rec) {{
    return rec.is_kernel &&
           (rec.klass == LC_GEMM || rec.klass == LC_ATTENTION ||
            csv_match(rec.name.c_str(), filter_csv));
}}

void nvbit_at_init() {{
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    if (const char* env = getenv("{_quote_cpp(analysis.filter_env)}")) {{
        filter_csv = env;
    }}
    if (const char* env = getenv("{_quote_cpp(analysis.window_env)}")) {{
        neighborhood_window = atoi(env);
        if (neighborhood_window < 1) neighborhood_window = 1;
        if (neighborhood_window > 8) neighborhood_window = 8;
    }}
    verbose = getenv("NVBPF_VERBOSE") != nullptr;
    full_names = getenv("NVBPF_FULL_NAMES") != nullptr;
    printf("[NVBPF {tool.banner}] Tool loaded\\n");
}}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {{
    if (!is_exit) return;
    api_event_counter++;

    if (cbid == API_CUDA_cuMemcpyPeer || cbid == API_CUDA_cuMemcpyPeer_ptds) {{
        size_t bytes = 0;
        if (cbid == API_CUDA_cuMemcpyPeer) {{
            bytes = ((cuMemcpyPeer_params*)params)->ByteCount;
        }} else {{
            bytes = ((cuMemcpyPeer_ptds_params*)params)->ByteCount;
        }}
        record_api_copy("api:cuMemcpyPeer", bytes);
        return;
    }}
    if (cbid == API_CUDA_cuMemcpyPeerAsync || cbid == API_CUDA_cuMemcpyPeerAsync_ptsz) {{
        size_t bytes = 0;
        if (cbid == API_CUDA_cuMemcpyPeerAsync) {{
            bytes = ((cuMemcpyPeerAsync_params*)params)->ByteCount;
        }} else {{
            bytes = ((cuMemcpyPeerAsync_ptsz_params*)params)->ByteCount;
        }}
        record_api_copy("api:cuMemcpyPeerAsync", bytes);
        return;
    }}
    if (cbid == API_CUDA_cuMemcpyDtoD_v2 ||
        cbid == API_CUDA_cuMemcpyDtoDAsync_v2 ||
        cbid == API_CUDA_cuMemcpyDtoD_v2_ptds ||
        cbid == API_CUDA_cuMemcpyDtoDAsync_v2_ptsz) {{
        size_t bytes = 0;
        const char* label = "api:cuMemcpyDtoD";
        if (cbid == API_CUDA_cuMemcpyDtoD_v2) {{
            bytes = ((cuMemcpyDtoD_v2_params*)params)->ByteCount;
        }} else if (cbid == API_CUDA_cuMemcpyDtoDAsync_v2) {{
            bytes = ((cuMemcpyDtoDAsync_v2_params*)params)->ByteCount;
            label = "api:cuMemcpyDtoDAsync";
        }} else if (cbid == API_CUDA_cuMemcpyDtoD_v2_ptds) {{
            bytes = ((cuMemcpyDtoD_v2_ptds_params*)params)->ByteCount;
        }} else {{
            bytes = ((cuMemcpyDtoDAsync_v2_ptsz_params*)params)->ByteCount;
            label = "api:cuMemcpyDtoDAsync";
        }}
        record_api_copy(label, bytes);
        return;
    }}

    if (!nvbpf_is_launch_event(cbid)) return;

    CUfunction func = nvbpf_get_launch_func(cbid, params);
    const char* func_name = nvbit_get_func_name(ctx, func);

    LaunchRecord rec{{}};
    rec.event_id = api_event_counter;
    rec.is_kernel = true;
    rec.klass = classify_name(func_name);
    rec.name = func_name;
    CUdevice dev = 0;
    if (cuCtxGetDevice(&dev) == CUDA_SUCCESS) {{
        rec.gpu = (int)dev;
    }}
    records.push_back(rec);
}}

void nvbit_at_term() {{
    size_t kernel_launches = 0;
    size_t gemm_launches = 0;
    size_t copy_events = 0;
    for (const auto& rec : records) {{
        if (rec.is_kernel) kernel_launches++;
        if (is_focus_gemm(rec)) gemm_launches++;
        if (rec.klass == LC_COPY) copy_events++;
    }}

    printf("[NVBPF {tool.banner}] launches=%zu focus_kernels=%zu copy_events=%zu\\n",
           kernel_launches, gemm_launches, copy_events);

    size_t ordinal = 0;
    for (size_t i = 0; i < records.size(); i++) {{
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

        if (verbose) {{
            printf("[NVBPF] gemm_neighborhood #%zu gpu=%d kernel=%s\\n",
                   ordinal, rec.gpu, rec.name.c_str());
        }}
        for (size_t j = lo; j <= hi; j++) {{
            if (j == i) {{
                if (verbose) {{
                    printf("          [0] %-11s %s\\n", class_name(records[j].klass),
                           records[j].name.c_str());
                }}
                continue;
            }}
            if (verbose) {{
                long rel = (long)j - (long)i;
                printf("         [%+ld] %-11s %s",
                       rel, class_name(records[j].klass), records[j].name.c_str());
                if (!records[j].is_kernel && records[j].bytes > 0) {{
                    printf(" bytes=%zu", records[j].bytes);
                }}
                printf("\\n");
            }}

            if (j < i) {{
                if (records[j].klass == LC_COPY) copy_before++;
                if (records[j].klass == LC_TRANSPOSE) transpose_before++;
                if (records[j].klass == LC_COPY || records[j].klass == LC_TRANSPOSE) {{
                    prep_before++;
                }}
            }} else if (j > i) {{
                if (records[j].klass == LC_EPILOGUE) epilogue_after++;
                if (records[j].klass == LC_ELEMENTWISE) elementwise_after++;
                if (records[j].klass == LC_REDUCTION) reduction_after++;
            }}
            if (records[j].klass == LC_ATTENTION) attention_neighbors++;
        }}

        if (verbose) {{
            printf("        summary: prep_before=%d copy_before=%d transpose_before=%d epilogue_after=%d elementwise_after=%d reduction_after=%d attention_neighbors=%d\\n",
                   prep_before, copy_before, transpose_before, epilogue_after,
                   elementwise_after, reduction_after, attention_neighbors);
            if (prep_before > 0) {{
                printf("        heuristic: GEMM is surrounded by explicit prep/copy work\\n");
            }}
            if (rec.klass == LC_ATTENTION || attention_neighbors > 0) {{
                printf("        heuristic: fused attention/FMHA kernels are present in the local neighborhood\\n");
            }}
            if (epilogue_after > 0 || elementwise_after > 0) {{
                printf("        heuristic: post-GEMM epilogue appears to run as separate kernels\\n");
            }} else {{
                printf("        heuristic: no obvious post-GEMM epilogue kernels in the local neighborhood; fused epilogue is possible\\n");
            }}
        }} else {{
            NeighborhoodAggregate* agg = find_aggregate(rec);
            if (agg == nullptr) {{
                NeighborhoodAggregate fresh{{}};
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
            }}
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
        }}
    }}

    if (!verbose) {{
        printf("[NVBPF {tool.banner}] unique_focus_kernels=%zu\\n", aggregates.size());
        for (const auto& agg : aggregates) {{
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
            if (agg.saw_fused_attention) {{
                printf("fused_attention");
                wrote = true;
            }}
            if (agg.saw_prep) {{
                printf("%sprep_copy", wrote ? "," : "");
                wrote = true;
            }}
            if (agg.saw_separate_epilogue) {{
                printf("%sseparate_epilogue", wrote ? "," : "");
                wrote = true;
            }}
            if (!wrote) {{
                printf("clean_neighborhood");
            }}
            printf("\\n");
        }}
    }}

    printf("[NVBPF {tool.banner}] Tool terminated\\n");
}}
"""


def render_host(tool: ToolSpec, out_dir: Path, core_relpath: str) -> str:
    if tool.gemm_wavefit is not None:
        return _render_gemm_wavefit_host(tool)
    if tool.gemm_orchestration is not None:
        return _render_gemm_orchestration_host(tool)

    prefix = _sanitize_ident(tool.name)
    has_device = _has_device_part(tool)
    has_opcode_checks = any(c.opcodes for c in tool.counters) or any(h.opcodes for h in tool.device_hooks)
    has_branch_checks = any(c.branches for c in tool.counters) or any(h.branches for h in tool.device_hooks)
    launch_state_decls = ""
    launch_state_init = ""
    if has_device:
        launch_state_decls = "\n".join(
            [
                "static pthread_mutex_t launch_mutex;",
                "static std::unordered_set<CUfunction> already_instrumented;",
            ]
        )
        launch_state_init = "    pthread_mutex_init(&launch_mutex, nullptr);"
    event_structs = _event_structs(tool)
    map_decls: list[str] = [_map_decl(map_spec) for map_spec in tool.maps]
    map_decls.extend(
        f'BPF_ARRAY({counter.name}, uint64_t, 1);  // {_counter_comment(counter)}'
        for counter in tool.counters
    )
    for event in tool.events:
        map_decls.append(
            f"BPF_RINGBUF({event.name}, {_event_type(event)}, {event.capacity});"
            + (f"  // {event.description}" if event.description else "")
        )
    externs: list[str] = []
    if tool.counters:
        externs.append(
            f'extern "C" __device__ __noinline__ void {prefix}_count_counter(int pred, uint64_t pcounter);'
        )
    events_by_name = {event.name: event for event in tool.events}
    counters_by_name = {counter.name for counter in tool.counters}
    for hook in tool.device_hooks:
        hook_render = render_custom_hook(hook, events_by_name, counters_by_name)
        externs.append(
            f'extern "C" __device__ __noinline__ void {prefix}_{hook.name}('
            + ", ".join(hook_render.signature_params)
            + ");"
        )

    reset_lines = "\n".join(
        [f"    {counter.name}.reset();" for counter in tool.counters]
        + [f"    {event.name}.reset();" for event in tool.events]
    )
    matcher_blocks = "\n\n".join(
        block for block in [
            _render_counter_matchers(tool),
            _render_hook_matchers(tool),
        ] if block
    )
    event_consumers = "".join(_render_event_consume(event) for event in tool.events)
    print_pairs = " ".join(f"{counter.name}=%lu" for counter in tool.counters)
    print_values = ", ".join(f"*{counter.name}.lookup(0)" for counter in tool.counters)
    device_instrumentation = _render_device_instrumentation(tool, prefix) if has_device else ""
    device_body = ""
    if has_device:
        counter_print = (
            f'        printf("        {print_pairs}\\n", {print_values});'
            if tool.counters
            else ""
        )
        device_body = f"""    if (!is_launch) return;
    CUfunction func = nvbpf_get_launch_func(cbid, params);
    const char* func_name = nvbit_get_func_name(ctx, func);

    if (!is_exit) {{
        bool match = kernel_name_filter.empty() ||
                     strstr(func_name, kernel_name_filter.c_str()) != nullptr;
        pthread_mutex_lock(&launch_mutex);
        if (match) {{
            instrument_function_if_needed(ctx, func);
            reset_state();
            nvbit_enable_instrumented(ctx, func, true);
        }} else {{
            nvbit_enable_instrumented(ctx, func, false);
        }}
        if (!match) pthread_mutex_unlock(&launch_mutex);
    }} else {{
        cudaDeviceSynchronize();
        if (!kernel_name_filter.empty() &&
            strstr(func_name, kernel_name_filter.c_str()) == nullptr) {{
            return;
        }}
        printf("[NVBPF] %s\\n", func_name);
{counter_print}
{event_consumers}
        pthread_mutex_unlock(&launch_mutex);
    }}"""
    elif not tool.api_traces:
        device_body = "    return;"

    return f"""/*
 * Auto-generated by nvbpf_py. Edit the Python spec, not this file.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>
{"#include <pthread.h>" if has_device else ""}
{"#include <unordered_set>" if has_device else ""}

#define NVBPF_NO_DEFAULT_CALLBACKS
#include "nvbpf.h"

{event_structs}

{chr(10).join(map_decls)}

{chr(10).join(externs)}

{launch_state_decls}
static std::string kernel_name_filter;
{"static bool verbose = false;" if tool.events else ""}
{_render_api_trace_statics(tool)}

{"static bool opcode_starts_with(const char* opcode, const char* prefix) { return strncmp(opcode, prefix, strlen(prefix)) == 0; }" if has_opcode_checks else ""}
{"static bool is_branch_opcode(const char* opcode) { return strncmp(opcode, \"BRA\", 3) == 0 || strncmp(opcode, \"JMP\", 3) == 0 || strncmp(opcode, \"JMX\", 3) == 0 || strncmp(opcode, \"BRX\", 3) == 0 || strncmp(opcode, \"CALL\", 4) == 0 || strncmp(opcode, \"RET\", 3) == 0 || strncmp(opcode, \"EXIT\", 4) == 0; }" if has_branch_checks else ""}

{matcher_blocks}

{"static void reset_state() {\n" + reset_lines + "\n}" if has_device else ""}

{("static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {\n"
  "    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);\n"
  "    related.push_back(func);\n"
  "    for (auto f : related) {\n"
  "        if (!already_instrumented.insert(f).second) continue;\n"
  "        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);\n"
  "        for (auto* instr : instrs) {\n"
  + device_instrumentation + "\n"
  "        }\n"
  "    }\n"
  "}") if has_device else ""}

void nvbit_at_init() {{
    setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
    {"setenv(\"CUDA_MANAGED_FORCE_DEVICE_ALLOC\", \"1\", 1);" if has_device else ""}
{launch_state_init}
    if (const char* env = getenv("{_quote_cpp(tool.kernel_filter_env)}")) {{
        kernel_name_filter = env;
    }}
    {"verbose = getenv(\"NVBPF_VERBOSE\") != nullptr;" if tool.events else ""}
    printf("[NVBPF {tool.banner}] Tool loaded\\n");
}}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {{
    bool is_launch = nvbpf_is_launch_event(cbid);
{_render_api_trace_body(tool)}
{device_body}
}}

void nvbit_at_term() {{
{_render_api_trace_term(tool)}
    printf("[NVBPF {tool.banner}] Tool terminated\\n");
}}
"""


def render_hook(tool: ToolSpec) -> str:
    if tool.gemm_wavefit is not None:
        return _render_gemm_wavefit_hook(tool)

    prefix = _sanitize_ident(tool.name)
    blocks: list[str] = [
        "/*",
        " * Auto-generated by nvbpf_py. Edit the Python spec, not this file.",
        " */",
        "",
        "#include <stdint.h>",
        '#include "nvbpf_helpers.h"',
        '#include "nvbpf_maps.h"',
        '#include "utils/utils.h"',
        "",
    ]
    event_structs = _event_structs(tool)
    if event_structs:
        blocks.append(event_structs)
        blocks.append("")

    if tool.counters:
        blocks.append(
            f'extern "C" __device__ __noinline__ void {prefix}_count_counter(int pred, uint64_t pcounter) {{'
        )
        blocks.extend(
            [
                "    const int active_mask = __ballot_sync(__activemask(), 1);",
                "    const int predicate_mask = __ballot_sync(__activemask(), pred);",
                "    const int laneid = get_laneid();",
                "    const int first_laneid = __ffs(active_mask) - 1;",
                "    if (first_laneid == laneid && __popc(predicate_mask) > 0) {",
                "        atomicAdd((unsigned long long*)pcounter, 1ULL);",
                "    }",
                "}",
                "",
            ]
        )

    events_by_name = {event.name: event for event in tool.events}
    counters_by_name = {counter.name for counter in tool.counters}
    for hook in tool.device_hooks:
        hook_render = render_custom_hook(hook, events_by_name, counters_by_name)
        blocks.append(
            f'extern "C" __device__ __noinline__ void {prefix}_{hook.name}('
            + ", ".join(hook_render.signature_params)
            + ") {"
        )
        blocks.append(hook_render.body)
        blocks.append("}")
        blocks.append("")

    return "\n".join(blocks).rstrip() + "\n"


def render_makefile(tool: ToolSpec, core_relpath: str) -> str:
    name = tool.name
    if _has_hook_file(tool):
        target_body = f"""{name}.so: {name}.o {name}_hooks.o $(NVBIT_PATH)/libnvbit.a
\t$(NVCC) -arch=$(ARCH) -O3 {name}.o {name}_hooks.o \\
\t        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

{name}.o: {name}.cu
\t$(NVCC) $(HOST_FLAGS) $< -o $@

{name}_hooks.o: {name}_hooks.cu
\t$(NVCC) $(HOOK_FLAGS) $< -o $@
"""
    else:
        target_body = f"""{name}.so: {name}.o $(NVBIT_PATH)/libnvbit.a
\t$(NVCC) -arch=$(ARCH) -O3 {name}.o \\
\t        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

{name}.o: {name}.cu
\t$(NVCC) $(HOST_FLAGS) $< -o $@
"""

    return f"""# Auto-generated by nvbpf_py. Edit the Python spec, not this file.

NVCC=nvcc -ccbin=$(CXX) -D_FORCE_INLINES
PTXAS=ptxas

NVCC_VER_REQ=10.1
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${{NVCC_VER}} >= $(NVCC_VER_REQ)" | bc)

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool!)
endif

PTXAS_VER_ADD_FLAG=12.3
PTXAS_VER=$(shell $(PTXAS) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
PTXAS_VER_CHECK=$(shell echo "${{PTXAS_VER}} >= $(PTXAS_VER_ADD_FLAG)" | bc)

ifeq ($(PTXAS_VER_CHECK), 0)
MAXRREGCOUNT_FLAG=-maxrregcount=24
else
MAXRREGCOUNT_FLAG=
endif

NVBIT_PATH={core_relpath}
INCLUDES=-I$(NVBIT_PATH)
LIBS=-L$(NVBIT_PATH) -lnvbit
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))
ARCH?=sm_89

HOST_FLAGS=-dc -c -std=c++17 $(INCLUDES) -Xptxas -cloning=no \\
           -Xcompiler -Wall -arch=$(ARCH) -O3 -Xcompiler -fPIC
HOOK_FLAGS=$(INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch \\
           --keep-device-functions -arch=$(ARCH) \\
           -Xcompiler -Wall -Xcompiler -fPIC -c

all: {name}.so

{target_body}
clean:
\trm -f *.so *.o
"""


def render_tool(tool: ToolSpec, out_dir: Path, repo_root: Path, *, include_makefile: bool = True) -> dict[str, str]:
    core_rel = (repo_root / "core").resolve()
    out_abs = out_dir.resolve()
    core_rel_for_make = os.path.relpath(core_rel, out_abs)
    name = tool.name
    rendered = {
        f"{name}.cu": render_host(tool, out_dir, core_rel_for_make),
    }
    if _has_hook_file(tool):
        rendered[f"{name}_hooks.cu"] = render_hook(tool)
    if include_makefile:
        rendered["Makefile"] = render_makefile(tool, core_rel_for_make)
    return rendered


def render_examples_makefile_block(tool: ToolSpec) -> str:
    name = tool.name
    if _has_hook_file(tool):
        body = f"""# ── {name} (nvbpf_py) ─────────────────────────────────────────────
{name}.so: {name}.o {name}_hooks.o $(NVBIT_PATH)/libnvbit.a
\t$(NVCC) -arch=$(ARCH) -O3 {name}.o {name}_hooks.o \\
\t        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

{name}.o: {name}.cu
\t$(NVCC) $(HOST_FLAGS) $< -o $@

{name}_hooks.o: {name}_hooks.cu
\t$(NVCC) $(HOOK_FLAGS) $< -o $@
"""
    else:
        body = f"""# ── {name} (nvbpf_py) ─────────────────────────────────────────────
{name}.so: {name}.o $(NVBIT_PATH)/libnvbit.a
\t$(NVCC) -arch=$(ARCH) -O3 {name}.o \\
\t        $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

{name}.o: {name}.cu
\t$(NVCC) $(HOST_FLAGS) $< -o $@
"""
    return body
