#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable
from xml.sax.saxutils import escape


@dataclass
class WavefitRow:
    launches: int
    kernel: str
    ctas: int
    fill_fraction: float
    used_sms: int
    total_sms: int
    regs: int
    smem_static: int
    smem_dynamic: int
    heuristic: str


@dataclass
class NeighborhoodRow:
    count: int
    klass: str
    kernel: str
    prep_min: int
    prep_max: int
    copy_min: int
    copy_max: int
    trans_min: int
    trans_max: int
    epi_min: int
    epi_max: int
    elem_min: int
    elem_max: int
    red_min: int
    red_max: int
    attn_min: int
    attn_max: int
    flags: tuple[str, ...]


@dataclass
class EpilogueRow:
    count: int
    klass: str
    kernel: str
    post_min: int
    post_max: int
    epi_min: int
    epi_max: int
    bias_min: int
    bias_max: int
    act_min: int
    act_max: int
    scale_min: int
    scale_max: int
    copy_min: int
    copy_max: int
    red_min: int
    red_max: int
    elem_min: int
    elem_max: int
    flags: tuple[str, ...]


@dataclass
class TailRow:
    launches: int
    kernel: str
    sites: int
    partial_pct: float
    low_pct: float
    dead_pct: float
    waste_pct: float
    avg_active: float
    math_partial_pct: float
    mem_partial_pct: float
    branch_partial_pct: float


@dataclass
class ReuseDistanceRow:
    launches: int
    kernel: str
    sampled: int
    hit_pct: float
    avg_gap: float
    max_gap: int
    miss: int
    h1: int
    h4: int
    h16: int
    h64: int
    hfar: int


@dataclass
class PipelineDepthRow:
    launches: int
    kernel: str
    producers: int
    consumers: int
    avg_gap: float
    max_gap: int
    burst: int
    stage: int
    overlap: int
    label: str


@dataclass
class TileLifetimeRow:
    launches: int
    kernel: str
    segments: int
    avg_life: float
    max_life: int
    avg_math: float
    t4: int
    t16: int
    t64: int
    t256: int
    tlong: int


@dataclass
class BankConflictRow:
    launches: int
    kernel: str
    shld: int
    shst: int
    ldm: int
    cp: int
    vec: int
    g2s: int
    score: int
    label: str


@dataclass
class RegisterPressureRow:
    launches: int
    kernel: str
    block: str
    regs: int
    occ: int
    local: int
    math: int
    score: float
    label: str


@dataclass
class CtaRoleRow:
    launches: int
    kernel: str
    ctas: int
    drop: int
    comp: int
    mem: int
    ctrl: int
    edge: int
    bal: int


@dataclass
class KernelSummaryRow:
    launches: int
    kernel: str
    kind: str
    regs: int
    smem_static: int
    smem_dynamic: int
    instrs: int
    tensor: int
    ffma: int
    ldmatrix: int
    cp_async: int
    branches: int
    loads: int
    stores: int
    active_sms: int


@dataclass
class SamplingRow:
    launches: int
    kernel: str
    loads: int
    stores: int
    sampled: int
    dropped: int
    sample_every: int
    addr_min: int
    addr_max: int


RANGE_RE = re.compile(r"^(?P<lo>\d+)(?:-(?P<hi>\d+))?$")
HEADER_RE = re.compile(r"\[NVBPF ([A-Z_]+)\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse NV-BPF tool logs and render SVG summaries."
    )
    parser.add_argument("--input", required=True, help="Path to a captured NV-BPF stdout log.")
    parser.add_argument(
        "--output-dir",
        default="tools/nvbpf_analysis/out",
        help="Directory for generated CSV/SVG outputs.",
    )
    parser.add_argument(
        "--tool",
        default="auto",
        choices=(
            "auto",
            "gemm_wavefit",
            "gemm_orchestration",
            "epilogue_fusion",
            "tail_fragment",
            "reuse_distance",
            "pipeline_depth",
            "tile_lifetime",
            "bank_conflict",
            "register_pressure",
            "cta_role",
            "kernel_summary",
            "sampling_mem_trace",
        ),
        help="Override tool-type detection.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=12,
        help="Maximum number of rows to include in the SVG view.",
    )
    return parser.parse_args()


def compact_label(text: str, limit: int = 34) -> str:
    if len(text) <= limit:
        return text
    keep = max(6, (limit - 3) // 2)
    return text[:keep] + "..." + text[-keep:]


def parse_range(text: str) -> tuple[int, int]:
    match = RANGE_RE.match(text)
    if not match:
        raise ValueError(f"invalid range token: {text!r}")
    lo = int(match.group("lo"))
    hi = int(match.group("hi") or match.group("lo"))
    return lo, hi


def parse_kv_section(section: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in re.finditer(r"([A-Za-z0-9_]+)=\s*([^\s]+)", section)}


def detect_tool(text: str) -> str:
    if "[NVBPF GEMM_WAVEFIT_TRACE]" in text:
        return "gemm_wavefit"
    if "[NVBPF GEMM_ORCHESTRATION_MAP]" in text:
        return "gemm_orchestration"
    if "[NVBPF EPILOGUE_FUSION_TRACE]" in text:
        return "epilogue_fusion"
    if "[NVBPF TAIL_FRAGMENT_TRACKER]" in text:
        return "tail_fragment"
    if "[NVBPF REUSE_DISTANCE_PROFILER]" in text:
        return "reuse_distance"
    if "[NVBPF PIPELINE_DEPTH_ESTIMATOR]" in text:
        return "pipeline_depth"
    if "[NVBPF TILE_LIFETIME_TRACKER]" in text:
        return "tile_lifetime"
    if "[NVBPF BANK_CONFLICT_SUSPICION]" in text:
        return "bank_conflict"
    if "[NVBPF REGISTER_PRESSURE_DISTORTION_METER]" in text:
        return "register_pressure"
    if "[NVBPF CTA_ROLE_CLASSIFIER]" in text:
        return "cta_role"
    if "[NVBPF KERNEL_SUMMARY]" in text:
        return "kernel_summary"
    if "[NVBPF SAMPLING_MEM_TRACE]" in text:
        return "sampling_mem_trace"
    raise RuntimeError("could not detect NV-BPF tool type from log")


def parse_wavefit(text: str) -> tuple[dict[str, int], list[WavefitRow]]:
    meta: dict[str, int] = {}
    rows: list[WavefitRow] = []
    for line in text.splitlines():
        if line.startswith("[NVBPF GEMM_WAVEFIT_TRACE] matched_launches="):
            info = parse_kv_section(line)
            meta["matched_launches"] = int(info["matched_launches"])
            meta["unique_kernels"] = int(info["unique_kernels"])
            continue
        stripped = line.strip()
        if not stripped.startswith("x") or "|" not in stripped:
            continue
        left, metrics, heuristic = [part.strip() for part in stripped.split("|", 2)]
        head = re.match(r"x(?P<count>\d+)\s+(?P<kernel>.+)", left)
        if not head:
            continue
        metric_values = parse_kv_section(metrics)
        used_sms, total_sms = metric_values["sms"].split("/")
        smem_static, smem_dynamic = metric_values["smem"].split("+")
        rows.append(
            WavefitRow(
                launches=int(head.group("count")),
                kernel=head.group("kernel").strip(),
                ctas=int(metric_values["ctas"]),
                fill_fraction=float(metric_values["fill"]),
                used_sms=int(used_sms),
                total_sms=int(total_sms),
                regs=int(metric_values["regs"]),
                smem_static=int(smem_static),
                smem_dynamic=int(smem_dynamic),
                heuristic=heuristic.strip(),
            )
        )
    return meta, rows


def parse_orchestration(text: str) -> tuple[dict[str, int], list[NeighborhoodRow]]:
    meta: dict[str, int] = {}
    rows: list[NeighborhoodRow] = []
    for line in text.splitlines():
        if line.startswith("[NVBPF GEMM_ORCHESTRATION_MAP] launches="):
            info = parse_kv_section(line)
            meta["launches"] = int(info["launches"])
            meta["focus_kernels"] = int(info["focus_kernels"])
            meta["copy_events"] = int(info["copy_events"])
            continue
        if line.startswith("[NVBPF GEMM_ORCHESTRATION_MAP] unique_focus_kernels="):
            info = parse_kv_section(line)
            meta["unique_focus_kernels"] = int(info["unique_focus_kernels"])
            continue
        stripped = line.strip()
        if not stripped.startswith("x") or "|" not in stripped:
            continue
        left, metrics, flags = [part.strip() for part in stripped.split("|", 2)]
        head = re.match(r"x(?P<count>\d+)\s+(?P<klass>\S+)\s+(?P<kernel>.+)", left)
        if not head:
            continue
        metric_values = parse_kv_section(metrics)
        prep_min, prep_max = parse_range(metric_values["prep"])
        copy_min, copy_max = parse_range(metric_values["copy"])
        trans_min, trans_max = parse_range(metric_values["trans"])
        epi_min, epi_max = parse_range(metric_values["epi"])
        elem_min, elem_max = parse_range(metric_values["elem"])
        red_min, red_max = parse_range(metric_values["red"])
        attn_min, attn_max = parse_range(metric_values["attn"])
        rows.append(
            NeighborhoodRow(
                count=int(head.group("count")),
                klass=head.group("klass"),
                kernel=head.group("kernel").strip(),
                prep_min=prep_min,
                prep_max=prep_max,
                copy_min=copy_min,
                copy_max=copy_max,
                trans_min=trans_min,
                trans_max=trans_max,
                epi_min=epi_min,
                epi_max=epi_max,
                elem_min=elem_min,
                elem_max=elem_max,
                red_min=red_min,
                red_max=red_max,
                attn_min=attn_min,
                attn_max=attn_max,
                flags=tuple(part.strip() for part in flags.split(",") if part.strip()),
            )
        )
    return meta, rows


def parse_epilogue(text: str) -> tuple[dict[str, int], list[EpilogueRow]]:
    meta: dict[str, int] = {}
    rows: list[EpilogueRow] = []
    for line in text.splitlines():
        if line.startswith("[NVBPF EPILOGUE_FUSION_TRACE] focus_kernels="):
            info = parse_kv_section(line)
            meta["focus_kernels"] = int(info["focus_kernels"])
            meta["fused_likely"] = int(info["fused_likely"])
            meta["separate_signals"] = int(info["separate_signals"])
            continue
        if line.startswith("[NVBPF EPILOGUE_FUSION_TRACE] unique_focus_kernels="):
            info = parse_kv_section(line)
            meta["unique_focus_kernels"] = int(info["unique_focus_kernels"])
            continue
        stripped = line.strip()
        if not stripped.startswith("x") or "|" not in stripped:
            continue
        left, metrics, flags = [part.strip() for part in stripped.split("|", 2)]
        head = re.match(r"x(?P<count>\d+)\s+(?P<klass>\S+)\s+(?P<kernel>.+)", left)
        if not head:
            continue
        metric_values = parse_kv_section(metrics)
        post_min, post_max = parse_range(metric_values["post"])
        epi_min, epi_max = parse_range(metric_values["epi"])
        bias_min, bias_max = parse_range(metric_values["bias"])
        act_min, act_max = parse_range(metric_values["act"])
        scale_min, scale_max = parse_range(metric_values["scale"])
        copy_min, copy_max = parse_range(metric_values["copy"])
        red_min, red_max = parse_range(metric_values["red"])
        elem_min, elem_max = parse_range(metric_values["elem"])
        rows.append(
            EpilogueRow(
                count=int(head.group("count")),
                klass=head.group("klass"),
                kernel=head.group("kernel").strip(),
                post_min=post_min,
                post_max=post_max,
                epi_min=epi_min,
                epi_max=epi_max,
                bias_min=bias_min,
                bias_max=bias_max,
                act_min=act_min,
                act_max=act_max,
                scale_min=scale_min,
                scale_max=scale_max,
                copy_min=copy_min,
                copy_max=copy_max,
                red_min=red_min,
                red_max=red_max,
                elem_min=elem_min,
                elem_max=elem_max,
                flags=tuple(part.strip() for part in flags.split(",") if part.strip()),
            )
        )
    return meta, rows


def parse_tail(text: str) -> tuple[dict[str, int], list[TailRow]]:
    meta: dict[str, int] = {}
    rows: list[TailRow] = []
    for line in text.splitlines():
        if line.startswith("[NVBPF TAIL_FRAGMENT_TRACKER] matched_launches="):
            info = parse_kv_section(line)
            meta["matched_launches"] = int(info["matched_launches"])
            meta["threshold"] = int(info["threshold"])
            meta["unique_kernels"] = int(info["unique_kernels"])
            continue
        stripped = line.strip()
        if not stripped.startswith("x") or "|" not in stripped:
            continue
        left, main_metrics, kind_metrics = [part.strip() for part in stripped.split("|", 2)]
        head = re.match(r"x(?P<count>\d+)\s+(?P<kernel>.+)", left)
        if not head:
            continue
        main = parse_kv_section(main_metrics.replace("%", ""))
        kind = parse_kv_section(kind_metrics.replace("%", ""))
        rows.append(
            TailRow(
                launches=int(head.group("count")),
                kernel=head.group("kernel").strip(),
                sites=int(main["sites"]),
                partial_pct=float(main["partial"]),
                low_pct=float(main["low"]),
                dead_pct=float(main["dead"]),
                waste_pct=float(main["waste"]),
                avg_active=float(main["avg"]),
                math_partial_pct=float(kind["math_p"]),
                mem_partial_pct=float(kind["mem_p"]),
                branch_partial_pct=float(kind["br_p"]),
            )
        )
    return meta, rows


def parse_reuse_distance(text: str) -> tuple[dict[str, int], list[ReuseDistanceRow]]:
    meta: dict[str, int] = {}
    rows: list[ReuseDistanceRow] = []
    for line in text.splitlines():
        if line.startswith("[NVBPF REUSE_DISTANCE_PROFILER] matched_launches="):
            info = parse_kv_section(line)
            meta["matched_launches"] = int(info["matched_launches"])
            meta["unique_kernels"] = int(info["unique_kernels"])
            meta["sample_every"] = int(info["sample_every"])
            meta["line_shift"] = int(info["line_shift"])
            continue
        stripped = line.strip()
        if not stripped.startswith("x") or "|" not in stripped:
            continue
        left, main_metrics, hist_metrics = [part.strip() for part in stripped.split("|", 2)]
        head = re.match(r"x(?P<count>\d+)\s+(?P<kernel>.+)", left)
        if not head:
            continue
        main = parse_kv_section(main_metrics.replace("%", ""))
        hist = parse_kv_section(hist_metrics)
        rows.append(
            ReuseDistanceRow(
                launches=int(head.group("count")),
                kernel=head.group("kernel").strip(),
                sampled=int(main["sampled"]),
                hit_pct=float(main["hit"]),
                avg_gap=float(main["avg_gap"]),
                max_gap=int(main["max"]),
                miss=int(hist["miss"]),
                h1=int(hist["h1"]),
                h4=int(hist["h4"]),
                h16=int(hist["h16"]),
                h64=int(hist["h64"]),
                hfar=int(hist["hfar"]),
            )
        )
    return meta, rows


def parse_pipeline_depth(text: str) -> tuple[dict[str, int], list[PipelineDepthRow]]:
    meta: dict[str, int] = {}
    rows: list[PipelineDepthRow] = []
    for line in text.splitlines():
        if line.startswith("[NVBPF PIPELINE_DEPTH_ESTIMATOR] matched_launches="):
            info = parse_kv_section(line)
            meta["matched_launches"] = int(info["matched_launches"])
            meta["unique_kernels"] = int(info["unique_kernels"])
            continue
        stripped = line.strip()
        if not stripped.startswith("x") or "|" not in stripped:
            continue
        left, metrics, label = [part.strip() for part in stripped.split("|", 2)]
        head = re.match(r"x(?P<count>\d+)\s+(?P<kernel>.+)", left)
        if not head:
            continue
        vals = parse_kv_section(metrics)
        rows.append(
            PipelineDepthRow(
                launches=int(head.group("count")),
                kernel=head.group("kernel").strip(),
                producers=int(vals["prod"]),
                consumers=int(vals["cons"]),
                avg_gap=float(vals["avg_gap"]),
                max_gap=int(vals["max"]),
                burst=int(vals["burst"]),
                stage=int(vals["stage"]),
                overlap=int(vals["overlap"]),
                label=label,
            )
        )
    return meta, rows


def parse_tile_lifetime(text: str) -> tuple[dict[str, int], list[TileLifetimeRow]]:
    meta: dict[str, int] = {}
    rows: list[TileLifetimeRow] = []
    for line in text.splitlines():
        if line.startswith("[NVBPF TILE_LIFETIME_TRACKER] matched_launches="):
            info = parse_kv_section(line)
            meta["matched_launches"] = int(info["matched_launches"])
            meta["unique_kernels"] = int(info["unique_kernels"])
            continue
        stripped = line.strip()
        if not stripped.startswith("x") or "|" not in stripped:
            continue
        left, main_metrics, hist_metrics = [part.strip() for part in stripped.split("|", 2)]
        head = re.match(r"x(?P<count>\d+)\s+(?P<kernel>.+)", left)
        if not head:
            continue
        main = parse_kv_section(main_metrics)
        hist = parse_kv_section(hist_metrics)
        rows.append(
            TileLifetimeRow(
                launches=int(head.group("count")),
                kernel=head.group("kernel").strip(),
                segments=int(main["seg"]),
                avg_life=float(main["avg_life"]),
                max_life=int(main["max"]),
                avg_math=float(main["avg_math"]),
                t4=int(hist["t4"]),
                t16=int(hist["t16"]),
                t64=int(hist["t64"]),
                t256=int(hist["t256"]),
                tlong=int(hist["tlong"]),
            )
        )
    return meta, rows


def parse_bank_conflict(text: str) -> tuple[dict[str, int], list[BankConflictRow]]:
    meta: dict[str, int] = {}
    rows: list[BankConflictRow] = []
    for line in text.splitlines():
        if line.startswith("[NVBPF BANK_CONFLICT_SUSPICION] matched_launches="):
            info = parse_kv_section(line)
            meta["matched_launches"] = int(info["matched_launches"])
            meta["unique_kernels"] = int(info["unique_kernels"])
            continue
        stripped = line.strip()
        if not stripped.startswith("x") or "|" not in stripped:
            continue
        left, metrics, label = [part.strip() for part in stripped.split("|", 2)]
        head = re.match(r"x(?P<count>\d+)\s+(?P<kernel>.+)", left)
        if not head:
            continue
        vals = parse_kv_section(metrics)
        rows.append(
            BankConflictRow(
                launches=int(head.group("count")),
                kernel=head.group("kernel").strip(),
                shld=int(vals["shld"]),
                shst=int(vals["shst"]),
                ldm=int(vals["ldm"]),
                cp=int(vals["cp"]),
                vec=int(vals["vec"]),
                g2s=int(vals["g2s"]),
                score=int(vals["score"]),
                label=label,
            )
        )
    return meta, rows


def parse_register_pressure(text: str) -> tuple[dict[str, int], list[RegisterPressureRow]]:
    meta: dict[str, int] = {}
    rows: list[RegisterPressureRow] = []
    for line in text.splitlines():
        if line.startswith("[NVBPF REGISTER_PRESSURE_DISTORTION_METER] matched_launches="):
            info = parse_kv_section(line)
            meta["matched_launches"] = int(info["matched_launches"])
            meta["unique_configs"] = int(info["unique_configs"])
            continue
        stripped = line.strip()
        if not stripped.startswith("x") or "|" not in stripped:
            continue
        parts = [part.strip() for part in stripped.split("|", 2)]
        if len(parts) != 3:
            continue
        left, metrics, label = parts
        head = re.match(r"x(?P<count>\d+)\s+(?P<kernel>.+?)\s+blk=(?P<blk>\S+)", left)
        if not head:
            continue
        vals = parse_kv_section(metrics)
        rows.append(
            RegisterPressureRow(
                launches=int(head.group("count")),
                kernel=head.group("kernel").strip(),
                block=head.group("blk"),
                regs=int(vals["regs"]),
                occ=int(vals["occ"]),
                local=int(vals["local"]),
                math=int(vals["math"]),
                score=float(vals["score"]),
                label=label,
            )
        )
    return meta, rows


def parse_cta_role(text: str) -> tuple[dict[str, int], list[CtaRoleRow]]:
    meta: dict[str, int] = {}
    rows: list[CtaRoleRow] = []
    for line in text.splitlines():
        if line.startswith("[NVBPF CTA_ROLE_CLASSIFIER] matched_launches="):
            info = parse_kv_section(line)
            meta["matched_launches"] = int(info["matched_launches"])
            meta["unique_kernels"] = int(info["unique_kernels"])
            continue
        stripped = line.strip()
        if not stripped.startswith("x") or "|" not in stripped:
            continue
        left, metrics = [part.strip() for part in stripped.split("|", 1)]
        head = re.match(r"x(?P<count>\d+)\s+(?P<kernel>.+)", left)
        if not head:
            continue
        vals = parse_kv_section(metrics)
        rows.append(
            CtaRoleRow(
                launches=int(head.group("count")),
                kernel=head.group("kernel").strip(),
                ctas=int(vals["ctas"]),
                drop=int(vals["drop"]),
                comp=int(vals["comp"]),
                mem=int(vals["mem"]),
                ctrl=int(vals["ctrl"]),
                edge=int(vals["edge"]),
                bal=int(vals["bal"]),
            )
        )
    return meta, rows


def parse_kernel_summary(text: str) -> tuple[dict[str, int], list[KernelSummaryRow]]:
    rows_by_kernel: dict[str, KernelSummaryRow] = {}
    lines = text.splitlines()
    i = 0
    launches = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("[NVBPF] "):
            i += 1
            continue
        head = re.match(r"\[NVBPF\] (?P<kernel>.+) \((?P<kind>.+)\)", line)
        if not head or i + 3 >= len(lines):
            i += 1
            continue
        launch_line = lines[i + 1].strip()
        instr_line = lines[i + 2].strip()
        mem_line = lines[i + 3].strip()
        launch_match = re.search(
            r"regs=(?P<regs>\d+) smem=(?P<static>\d+)\+(?P<dynamic>\d+)", launch_line
        )
        instr = parse_kv_section(instr_line)
        mem = parse_kv_section(mem_line.replace("mem:", "").strip())
        kernel = head.group("kernel").strip()
        kind = head.group("kind").strip()
        launches += 1
        if kernel not in rows_by_kernel:
            rows_by_kernel[kernel] = KernelSummaryRow(
                launches=0,
                kernel=kernel,
                kind=kind,
                regs=0,
                smem_static=0,
                smem_dynamic=0,
                instrs=0,
                tensor=0,
                ffma=0,
                ldmatrix=0,
                cp_async=0,
                branches=0,
                loads=0,
                stores=0,
                active_sms=0,
            )
        row = rows_by_kernel[kernel]
        row.launches += 1
        if launch_match:
            row.regs = max(row.regs, int(launch_match.group("regs")))
            row.smem_static = max(row.smem_static, int(launch_match.group("static")))
            row.smem_dynamic = max(row.smem_dynamic, int(launch_match.group("dynamic")))
        row.instrs += int(instr["instrs"])
        row.tensor += int(instr["tensor"])
        row.ffma += int(instr["ffma"])
        row.ldmatrix += int(instr["ldmatrix"])
        row.cp_async += int(instr["cp_async"])
        row.branches += int(instr["branches"])
        row.loads += int(mem["loads"])
        row.stores += int(mem["stores"])
        row.active_sms = max(row.active_sms, int(mem["active_sms"]))
        i += 4
    return {"matched_launches": launches, "unique_kernels": len(rows_by_kernel)}, list(rows_by_kernel.values())


def parse_sampling(text: str) -> tuple[dict[str, int], list[SamplingRow]]:
    rows_by_kernel: dict[str, SamplingRow] = {}
    lines = text.splitlines()
    i = 0
    launches = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("[NVBPF] "):
            i += 1
            continue
        kernel = line[len("[NVBPF] ") :].strip()
        if i + 2 >= len(lines):
            i += 1
            continue
        stats = parse_kv_section(lines[i + 1].strip())
        addr_match = re.search(
            r"addr_window=\[0x(?P<lo>[0-9a-fA-F]+), 0x(?P<hi>[0-9a-fA-F]+)\]",
            lines[i + 2].strip(),
        )
        launches += 1
        if kernel not in rows_by_kernel:
            rows_by_kernel[kernel] = SamplingRow(
                launches=0,
                kernel=kernel,
                loads=0,
                stores=0,
                sampled=0,
                dropped=0,
                sample_every=int(stats["sample_every"]),
                addr_min=(int(addr_match.group("lo"), 16) if addr_match else 0),
                addr_max=(int(addr_match.group("hi"), 16) if addr_match else 0),
            )
        row = rows_by_kernel[kernel]
        row.launches += 1
        row.loads += int(stats["loads"])
        row.stores += int(stats["stores"])
        row.sampled += int(stats["sampled"])
        row.dropped += int(stats["dropped"])
        row.sample_every = int(stats["sample_every"])
        if addr_match:
            row.addr_min = min(row.addr_min or int(addr_match.group("lo"), 16), int(addr_match.group("lo"), 16))
            row.addr_max = max(row.addr_max, int(addr_match.group("hi"), 16))
        i += 3
    return {"matched_launches": launches, "unique_kernels": len(rows_by_kernel)}, list(rows_by_kernel.values())


def sort_rows(tool: str, rows: list[object]) -> list[object]:
    if tool == "gemm_wavefit":
        return sorted(rows, key=lambda row: (row.fill_fraction, -row.launches, row.kernel))  # type: ignore[attr-defined]
    if tool == "gemm_orchestration":
        return sorted(rows, key=lambda row: (-row.count, row.kernel))  # type: ignore[attr-defined]
    if tool == "epilogue_fusion":
        return sorted(rows, key=lambda row: (-row.count, row.kernel))  # type: ignore[attr-defined]
    if tool == "tail_fragment":
        return sorted(rows, key=lambda row: (-row.waste_pct, -row.partial_pct, row.kernel))  # type: ignore[attr-defined]
    if tool == "reuse_distance":
        return sorted(rows, key=lambda row: (-row.hit_pct, row.avg_gap, row.kernel))  # type: ignore[attr-defined]
    if tool == "pipeline_depth":
        return sorted(rows, key=lambda row: (-row.stage, -row.overlap, row.kernel))  # type: ignore[attr-defined]
    if tool == "tile_lifetime":
        return sorted(rows, key=lambda row: (-row.avg_life, -row.avg_math, row.kernel))  # type: ignore[attr-defined]
    if tool == "bank_conflict":
        return sorted(rows, key=lambda row: (-row.score, row.kernel))  # type: ignore[attr-defined]
    if tool == "register_pressure":
        return sorted(rows, key=lambda row: (-row.score, -row.regs, row.kernel))  # type: ignore[attr-defined]
    if tool == "cta_role":
        return sorted(rows, key=lambda row: (-row.edge, -row.ctas, row.kernel))  # type: ignore[attr-defined]
    if tool == "kernel_summary":
        return sorted(rows, key=lambda row: (-row.instrs, row.kernel))  # type: ignore[attr-defined]
    if tool == "sampling_mem_trace":
        return sorted(rows, key=lambda row: (-row.sampled, row.kernel))  # type: ignore[attr-defined]
    return rows


def color_for_ratio(ratio: float) -> tuple[str, str]:
    ratio = max(0.0, min(1.0, ratio))
    start = (240, 244, 248)
    end = (26, 115, 232)
    rgb = tuple(int(start[i] + (end[i] - start[i]) * ratio) for i in range(3))
    fill = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    text = "#111111" if ratio < 0.55 else "#ffffff"
    return fill, text


def svg_heatmap_panel(
    *,
    x: int,
    y: int,
    title: str,
    row_labels: list[str],
    columns: list[str],
    numeric_values: list[list[float]],
    display_values: list[list[str]],
    panel_width: int,
) -> tuple[list[str], int]:
    label_width = 260
    cell_w = max(72, min(110, (panel_width - label_width - 20) // max(1, len(columns))))
    row_h = 28
    title_h = 28
    panel_h = title_h + 30 + len(row_labels) * row_h + 20
    parts = [
        f'<text x="{x}" y="{y + 20}" font-size="18" font-weight="700" fill="#111111">{escape(title)}</text>'
    ]
    col_max = []
    for col_idx in range(len(columns)):
        values = [row[col_idx] for row in numeric_values]
        col_max.append(max(values) if values else 0.0)

    table_y = y + title_h + 12
    for idx, column in enumerate(columns):
        cx = x + label_width + idx * cell_w + cell_w / 2
        parts.append(
            f'<text x="{cx:.1f}" y="{table_y - 6}" font-size="11" text-anchor="middle" fill="#333333">{escape(column)}</text>'
        )
    for row_idx, row_label in enumerate(row_labels):
        ry = table_y + row_idx * row_h
        parts.append(
            f'<text x="{x + label_width - 8}" y="{ry + 19}" font-size="11" text-anchor="end" fill="#222222">{escape(compact_label(row_label, 40))}</text>'
        )
        for col_idx, display in enumerate(display_values[row_idx]):
            value = numeric_values[row_idx][col_idx]
            denom = col_max[col_idx]
            ratio = 0.0 if denom <= 0 else value / denom
            fill, fg = color_for_ratio(ratio)
            rx = x + label_width + col_idx * cell_w
            parts.append(
                f'<rect x="{rx}" y="{ry}" width="{cell_w - 2}" height="{row_h - 2}" fill="{fill}" rx="4" ry="4"/>'
            )
            parts.append(
                f'<text x="{rx + (cell_w - 2) / 2:.1f}" y="{ry + 18}" font-size="11" text-anchor="middle" fill="{fg}">{escape(display)}</text>'
            )
    return parts, panel_h


def svg_document(width: int, height: int, parts: Iterable[str]) -> str:
    body = "\n".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        f"{body}\n"
        "</svg>\n"
    )


def render_panels_svg(title: str, subtitles: list[str], panels: list[tuple[str, list[str], list[str], list[list[float]], list[list[str]]]]) -> str:
    width = 1400
    parts = [
        f'<text x="28" y="34" font-size="28" font-weight="700" fill="#111111">{escape(title)}</text>'
    ]
    y = 60
    for line in subtitles:
        parts.append(f'<text x="28" y="{y}" font-size="13" fill="#444444">{escape(line)}</text>')
        y += 18
    y += 10
    for panel_title, row_labels, columns, numeric, display in panels:
        panel_parts, panel_h = svg_heatmap_panel(
            x=28,
            y=y,
            title=panel_title,
            row_labels=row_labels,
            columns=columns,
            numeric_values=numeric,
            display_values=display,
            panel_width=width - 56,
        )
        parts.extend(panel_parts)
        y += panel_h + 20
    return svg_document(width, y + 20, parts)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rows_to_dicts(rows: list[object]) -> list[dict[str, object]]:
    return [row.__dict__.copy() for row in rows]


def build_wavefit_outputs(meta: dict[str, int], rows: list[WavefitRow]) -> tuple[list[dict[str, object]], str]:
    csv_rows = rows_to_dicts(rows)
    row_labels = [row.kernel for row in rows]
    columns = ["launches", "ctas", "fill", "sms_used", "sms_util", "regs", "smem"]
    numeric = [
        [
            float(row.launches),
            float(row.ctas),
            row.fill_fraction,
            float(row.used_sms),
            (float(row.used_sms) / float(row.total_sms)) if row.total_sms else 0.0,
            float(row.regs),
            float(row.smem_static + row.smem_dynamic),
        ]
        for row in rows
    ]
    display = [
        [
            str(row.launches),
            str(row.ctas),
            f"{row.fill_fraction:.3f}",
            str(row.used_sms),
            f"{(row.used_sms / row.total_sms):.3f}" if row.total_sms else "0.000",
            str(row.regs),
            f"{row.smem_static}+{row.smem_dynamic}",
        ]
        for row in rows
    ]
    heuristics_numeric = [[1.0] for _ in rows]
    heuristics_display = [[row.heuristic] for row in rows]
    svg = render_panels_svg(
        "GEMM Wave-Fit Summary",
        [
            f"matched_launches={meta.get('matched_launches', 0)} unique_kernels={meta.get('unique_kernels', len(rows))}",
            "Columns show launch count, CTA count, fill fraction, used SMs, SM utilization, registers, and shared memory.",
        ],
        [
            ("Kernel Metrics", row_labels, columns, numeric, display),
            ("Heuristic Labels", row_labels, ["heuristic"], heuristics_numeric, heuristics_display),
        ],
    )
    return csv_rows, svg


def build_orchestration_outputs(meta: dict[str, int], rows: list[NeighborhoodRow]) -> tuple[list[dict[str, object]], str]:
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "count": row.count,
                "klass": row.klass,
                "kernel": row.kernel,
                "prep_min": row.prep_min,
                "prep_max": row.prep_max,
                "copy_min": row.copy_min,
                "copy_max": row.copy_max,
                "trans_min": row.trans_min,
                "trans_max": row.trans_max,
                "epi_min": row.epi_min,
                "epi_max": row.epi_max,
                "elem_min": row.elem_min,
                "elem_max": row.elem_max,
                "red_min": row.red_min,
                "red_max": row.red_max,
                "attn_min": row.attn_min,
                "attn_max": row.attn_max,
                "flags": ",".join(row.flags),
            }
        )
    row_labels = [f"{row.klass} {row.kernel}" for row in rows]
    count_cols = ["count", "prep_max", "copy_max", "trans_max", "epi_max", "elem_max", "red_max", "attn_max"]
    numeric = [
        [
            float(row.count),
            float(row.prep_max),
            float(row.copy_max),
            float(row.trans_max),
            float(row.epi_max),
            float(row.elem_max),
            float(row.red_max),
            float(row.attn_max),
        ]
        for row in rows
    ]
    display = [
        [
            str(row.count),
            f"{row.prep_min}-{row.prep_max}" if row.prep_min != row.prep_max else str(row.prep_max),
            f"{row.copy_min}-{row.copy_max}" if row.copy_min != row.copy_max else str(row.copy_max),
            f"{row.trans_min}-{row.trans_max}" if row.trans_min != row.trans_max else str(row.trans_max),
            f"{row.epi_min}-{row.epi_max}" if row.epi_min != row.epi_max else str(row.epi_max),
            f"{row.elem_min}-{row.elem_max}" if row.elem_min != row.elem_max else str(row.elem_max),
            f"{row.red_min}-{row.red_max}" if row.red_min != row.red_max else str(row.red_max),
            f"{row.attn_min}-{row.attn_max}" if row.attn_min != row.attn_max else str(row.attn_max),
        ]
        for row in rows
    ]
    flag_names = ["fused_attention", "prep_copy", "separate_epilogue", "clean_neighborhood"]
    flag_numeric = []
    flag_display = []
    for row in rows:
        row_set = set(row.flags)
        flag_numeric.append([1.0 if flag in row_set else 0.0 for flag in flag_names])
        flag_display.append(["yes" if flag in row_set else "" for flag in flag_names])
    svg = render_panels_svg(
        "GEMM Orchestration Summary",
        [
            f"launches={meta.get('launches', 0)} focus_kernels={meta.get('focus_kernels', 0)} copy_events={meta.get('copy_events', 0)}",
            "The first panel shows neighborhood count ranges; the second highlights fused-attention, prep/copy, and separate-epilogue signals.",
        ],
        [
            ("Neighborhood Counts", row_labels, count_cols, numeric, display),
            ("Behavior Flags", row_labels, flag_names, flag_numeric, flag_display),
        ],
    )
    return csv_rows, svg


def build_epilogue_outputs(meta: dict[str, int], rows: list[EpilogueRow]) -> tuple[list[dict[str, object]], str]:
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "count": row.count,
                "klass": row.klass,
                "kernel": row.kernel,
                "post_min": row.post_min,
                "post_max": row.post_max,
                "epi_min": row.epi_min,
                "epi_max": row.epi_max,
                "bias_min": row.bias_min,
                "bias_max": row.bias_max,
                "act_min": row.act_min,
                "act_max": row.act_max,
                "scale_min": row.scale_min,
                "scale_max": row.scale_max,
                "copy_min": row.copy_min,
                "copy_max": row.copy_max,
                "red_min": row.red_min,
                "red_max": row.red_max,
                "elem_min": row.elem_min,
                "elem_max": row.elem_max,
                "flags": ",".join(row.flags),
            }
        )
    row_labels = [f"{row.klass} {row.kernel}" for row in rows]
    count_cols = ["count", "post_max", "epi_max", "bias_max", "act_max", "scale_max", "copy_max", "red_max", "elem_max"]
    numeric = [
        [
            float(row.count),
            float(row.post_max),
            float(row.epi_max),
            float(row.bias_max),
            float(row.act_max),
            float(row.scale_max),
            float(row.copy_max),
            float(row.red_max),
            float(row.elem_max),
        ]
        for row in rows
    ]
    display = [
        [
            str(row.count),
            f"{row.post_min}-{row.post_max}" if row.post_min != row.post_max else str(row.post_max),
            f"{row.epi_min}-{row.epi_max}" if row.epi_min != row.epi_max else str(row.epi_max),
            f"{row.bias_min}-{row.bias_max}" if row.bias_min != row.bias_max else str(row.bias_max),
            f"{row.act_min}-{row.act_max}" if row.act_min != row.act_max else str(row.act_max),
            f"{row.scale_min}-{row.scale_max}" if row.scale_min != row.scale_max else str(row.scale_max),
            f"{row.copy_min}-{row.copy_max}" if row.copy_min != row.copy_max else str(row.copy_max),
            f"{row.red_min}-{row.red_max}" if row.red_min != row.red_max else str(row.red_max),
            f"{row.elem_min}-{row.elem_max}" if row.elem_min != row.elem_max else str(row.elem_max),
        ]
        for row in rows
    ]
    flag_names = [
        "attention_core",
        "fused_likely",
        "separate_bias",
        "separate_activation",
        "separate_scale",
        "separate_epilogue",
        "copyout_after",
        "reduction_tail",
    ]
    flag_numeric = []
    flag_display = []
    for row in rows:
        row_set = set(row.flags)
        flag_numeric.append([1.0 if flag in row_set else 0.0 for flag in flag_names])
        flag_display.append(["yes" if flag in row_set else "" for flag in flag_names])
    svg = render_panels_svg(
        "Epilogue Fusion Summary",
        [
            f"focus_kernels={meta.get('focus_kernels', 0)} fused_likely={meta.get('fused_likely', 0)} separate_signals={meta.get('separate_signals', 0)}",
            "Counts show post-kernel work windows; flags show where likely fusion or separate epilogue signals were observed.",
        ],
        [
            ("Post-Kernel Counts", row_labels, count_cols, numeric, display),
            ("Fusion Flags", row_labels, flag_names, flag_numeric, flag_display),
        ],
    )
    return csv_rows, svg


def build_tail_outputs(meta: dict[str, int], rows: list[TailRow]) -> tuple[list[dict[str, object]], str]:
    csv_rows = rows_to_dicts(rows)
    row_labels = [row.kernel for row in rows]
    columns = ["launches", "sites", "partial%", "low%", "dead%", "waste%", "avg_active", "math_p%", "mem_p%", "br_p%"]
    numeric = [
        [
            float(row.launches),
            float(row.sites),
            row.partial_pct,
            row.low_pct,
            row.dead_pct,
            row.waste_pct,
            row.avg_active,
            row.math_partial_pct,
            row.mem_partial_pct,
            row.branch_partial_pct,
        ]
        for row in rows
    ]
    display = [
        [
            str(row.launches),
            str(row.sites),
            f"{row.partial_pct:.2f}",
            f"{row.low_pct:.2f}",
            f"{row.dead_pct:.2f}",
            f"{row.waste_pct:.2f}",
            f"{row.avg_active:.2f}",
            f"{row.math_partial_pct:.2f}",
            f"{row.mem_partial_pct:.2f}",
            f"{row.branch_partial_pct:.2f}",
        ]
        for row in rows
    ]
    svg = render_panels_svg(
        "Tail Fragment Summary",
        [
            f"matched_launches={meta.get('matched_launches', 0)} threshold={meta.get('threshold', 0)} unique_kernels={meta.get('unique_kernels', len(rows))}",
            "Higher partial/low/dead/waste percentages point to stronger tail and edge inefficiency.",
        ],
        [("Tail Metrics", row_labels, columns, numeric, display)],
    )
    return csv_rows, svg


def build_reuse_distance_outputs(meta: dict[str, int], rows: list[ReuseDistanceRow]) -> tuple[list[dict[str, object]], str]:
    csv_rows = rows_to_dicts(rows)
    row_labels = [row.kernel for row in rows]
    numeric = [
        [
            float(row.launches),
            float(row.sampled),
            row.hit_pct,
            row.avg_gap,
            float(row.max_gap),
            float(row.miss),
            float(row.h1 + row.h4 + row.h16 + row.h64 + row.hfar),
        ]
        for row in rows
    ]
    display = [
        [
            str(row.launches),
            str(row.sampled),
            f"{row.hit_pct:.2f}",
            f"{row.avg_gap:.2f}",
            str(row.max_gap),
            str(row.miss),
            str(row.h1 + row.h4 + row.h16 + row.h64 + row.hfar),
        ]
        for row in rows
    ]
    hist_numeric = [
        [float(row.h1), float(row.h4), float(row.h16), float(row.h64), float(row.hfar)]
        for row in rows
    ]
    hist_display = [
        [str(row.h1), str(row.h4), str(row.h16), str(row.h64), str(row.hfar)]
        for row in rows
    ]
    svg = render_panels_svg(
        "Reuse Distance Summary",
        [
            f"matched_launches={meta.get('matched_launches', 0)} unique_kernels={meta.get('unique_kernels', len(rows))} sample_every={meta.get('sample_every', 0)} line_shift={meta.get('line_shift', 0)}",
            "Higher hit% and lower avg_gap indicate tighter sampled memory-line reuse.",
        ],
        [
            ("Reuse Metrics", row_labels, ["launches", "sampled", "hit%", "avg_gap", "max_gap", "miss", "hits"], numeric, display),
            ("Hit Buckets", row_labels, ["h1", "h4", "h16", "h64", "hfar"], hist_numeric, hist_display),
        ],
    )
    return csv_rows, svg


def build_pipeline_depth_outputs(meta: dict[str, int], rows: list[PipelineDepthRow]) -> tuple[list[dict[str, object]], str]:
    csv_rows = rows_to_dicts(rows)
    row_labels = [row.kernel for row in rows]
    numeric = [
        [
            float(row.launches),
            float(row.producers),
            float(row.consumers),
            row.avg_gap,
            float(row.max_gap),
            float(row.burst),
            float(row.stage),
            float(row.overlap),
        ]
        for row in rows
    ]
    display = [
        [
            str(row.launches),
            str(row.producers),
            str(row.consumers),
            f"{row.avg_gap:.2f}",
            str(row.max_gap),
            str(row.burst),
            str(row.stage),
            str(row.overlap),
        ]
        for row in rows
    ]
    label_numeric = [[float(row.stage)] for row in rows]
    label_display = [[row.label] for row in rows]
    svg = render_panels_svg(
        "Pipeline Depth Summary",
        [
            f"matched_launches={meta.get('matched_launches', 0)} unique_kernels={meta.get('unique_kernels', len(rows))}",
            "Stage and overlap estimate how deeply producer and consumer work is pipelined.",
        ],
        [
            ("Pipeline Metrics", row_labels, ["launches", "prod", "cons", "avg_gap", "max_gap", "burst", "stage", "overlap"], numeric, display),
            ("Pipeline Label", row_labels, ["label"], label_numeric, label_display),
        ],
    )
    return csv_rows, svg


def build_tile_lifetime_outputs(meta: dict[str, int], rows: list[TileLifetimeRow]) -> tuple[list[dict[str, object]], str]:
    csv_rows = rows_to_dicts(rows)
    row_labels = [row.kernel for row in rows]
    numeric = [
        [
            float(row.launches),
            float(row.segments),
            row.avg_life,
            float(row.max_life),
            row.avg_math,
        ]
        for row in rows
    ]
    display = [
        [
            str(row.launches),
            str(row.segments),
            f"{row.avg_life:.2f}",
            str(row.max_life),
            f"{row.avg_math:.2f}",
        ]
        for row in rows
    ]
    hist_numeric = [
        [float(row.t4), float(row.t16), float(row.t64), float(row.t256), float(row.tlong)]
        for row in rows
    ]
    hist_display = [
        [str(row.t4), str(row.t16), str(row.t64), str(row.t256), str(row.tlong)]
        for row in rows
    ]
    svg = render_panels_svg(
        "Tile Lifetime Summary",
        [
            f"matched_launches={meta.get('matched_launches', 0)} unique_kernels={meta.get('unique_kernels', len(rows))}",
            "Longer average lifetimes and higher avg_math suggest more compute per tile-residency window.",
        ],
        [
            ("Lifetime Metrics", row_labels, ["launches", "segments", "avg_life", "max_life", "avg_math"], numeric, display),
            ("Lifetime Buckets", row_labels, ["t4", "t16", "t64", "t256", "tlong"], hist_numeric, hist_display),
        ],
    )
    return csv_rows, svg


def build_bank_conflict_outputs(meta: dict[str, int], rows: list[BankConflictRow]) -> tuple[list[dict[str, object]], str]:
    csv_rows = rows_to_dicts(rows)
    row_labels = [row.kernel for row in rows]
    numeric = [
        [
            float(row.launches),
            float(row.shld),
            float(row.shst),
            float(row.ldm),
            float(row.cp),
            float(row.vec),
            float(row.g2s),
            float(row.score),
        ]
        for row in rows
    ]
    display = [
        [
            str(row.launches),
            str(row.shld),
            str(row.shst),
            str(row.ldm),
            str(row.cp),
            str(row.vec),
            str(row.g2s),
            str(row.score),
        ]
        for row in rows
    ]
    label_numeric = [[float(row.score)] for row in rows]
    label_display = [[row.label] for row in rows]
    svg = render_panels_svg(
        "Bank Conflict Suspicion Summary",
        [
            f"matched_launches={meta.get('matched_launches', 0)} unique_kernels={meta.get('unique_kernels', len(rows))}",
            "This is a heuristic score based on shared-memory instruction mix, not a direct bank-conflict counter.",
        ],
        [
            ("Shared-Memory Mix", row_labels, ["launches", "shld", "shst", "ldm", "cp", "vec", "g2s", "score"], numeric, display),
            ("Suspicion Label", row_labels, ["label"], label_numeric, label_display),
        ],
    )
    return csv_rows, svg


def build_register_pressure_outputs(meta: dict[str, int], rows: list[RegisterPressureRow]) -> tuple[list[dict[str, object]], str]:
    csv_rows = rows_to_dicts(rows)
    row_labels = [f"{row.kernel} [{row.block}]" for row in rows]
    numeric = [
        [
            float(row.launches),
            float(row.regs),
            float(row.occ),
            float(row.local),
            float(row.math),
            row.score,
        ]
        for row in rows
    ]
    display = [
        [
            str(row.launches),
            str(row.regs),
            str(row.occ),
            str(row.local),
            str(row.math),
            f"{row.score:.2f}",
        ]
        for row in rows
    ]
    label_numeric = [[row.score] for row in rows]
    label_display = [[row.label] for row in rows]
    svg = render_panels_svg(
        "Register Pressure Distortion Summary",
        [
            f"matched_launches={meta.get('matched_launches', 0)} unique_configs={meta.get('unique_configs', len(rows))}",
            "Higher scores indicate stronger suspicion that register use and local traffic are distorting residency.",
        ],
        [
            ("Pressure Metrics", row_labels, ["launches", "regs", "occ", "local", "math", "score"], numeric, display),
            ("Pressure Label", row_labels, ["label"], label_numeric, label_display),
        ],
    )
    return csv_rows, svg


def build_cta_role_outputs(meta: dict[str, int], rows: list[CtaRoleRow]) -> tuple[list[dict[str, object]], str]:
    csv_rows = rows_to_dicts(rows)
    row_labels = [row.kernel for row in rows]
    numeric = [
        [
            float(row.launches),
            float(row.ctas),
            float(row.drop),
            float(row.comp),
            float(row.mem),
            float(row.ctrl),
            float(row.edge),
            float(row.bal),
        ]
        for row in rows
    ]
    display = [
        [
            str(row.launches),
            str(row.ctas),
            str(row.drop),
            str(row.comp),
            str(row.mem),
            str(row.ctrl),
            str(row.edge),
            str(row.bal),
        ]
        for row in rows
    ]
    svg = render_panels_svg(
        "CTA Role Summary",
        [
            f"matched_launches={meta.get('matched_launches', 0)} unique_kernels={meta.get('unique_kernels', len(rows))}",
            "The role counts show how sampled CTAs cluster into compute, memory, control, edge, or balanced behavior.",
        ],
        [("CTA Role Counts", row_labels, ["launches", "ctas", "drop", "comp", "mem", "ctrl", "edge", "bal"], numeric, display)],
    )
    return csv_rows, svg


def build_kernel_summary_outputs(meta: dict[str, int], rows: list[KernelSummaryRow]) -> tuple[list[dict[str, object]], str]:
    csv_rows = rows_to_dicts(rows)
    row_labels = [f"{row.kind}: {row.kernel}" for row in rows]
    columns = ["launches", "instrs", "tensor", "ffma", "ldmatrix", "cp_async", "branches", "loads", "stores", "active_sms", "regs", "smem"]
    numeric = [
        [
            float(row.launches),
            float(row.instrs),
            float(row.tensor),
            float(row.ffma),
            float(row.ldmatrix),
            float(row.cp_async),
            float(row.branches),
            float(row.loads),
            float(row.stores),
            float(row.active_sms),
            float(row.regs),
            float(row.smem_static + row.smem_dynamic),
        ]
        for row in rows
    ]
    display = [
        [
            str(row.launches),
            str(row.instrs),
            str(row.tensor),
            str(row.ffma),
            str(row.ldmatrix),
            str(row.cp_async),
            str(row.branches),
            str(row.loads),
            str(row.stores),
            str(row.active_sms),
            str(row.regs),
            f"{row.smem_static}+{row.smem_dynamic}",
        ]
        for row in rows
    ]
    svg = render_panels_svg(
        "Kernel Summary Metrics",
        [
            f"matched_launches={meta.get('matched_launches', 0)} unique_kernels={meta.get('unique_kernels', len(rows))}",
            "These rows aggregate repeated kernel launches into per-kernel totals and maxima.",
        ],
        [("Kernel Metrics", row_labels, columns, numeric, display)],
    )
    return csv_rows, svg


def build_sampling_outputs(meta: dict[str, int], rows: list[SamplingRow]) -> tuple[list[dict[str, object]], str]:
    csv_rows = rows_to_dicts(rows)
    row_labels = [row.kernel for row in rows]
    columns = ["launches", "loads", "stores", "sampled", "dropped", "sample_every", "addr_span"]
    numeric = [
        [
            float(row.launches),
            float(row.loads),
            float(row.stores),
            float(row.sampled),
            float(row.dropped),
            float(row.sample_every),
            float(max(0, row.addr_max - row.addr_min)),
        ]
        for row in rows
    ]
    display = [
        [
            str(row.launches),
            str(row.loads),
            str(row.stores),
            str(row.sampled),
            str(row.dropped),
            str(row.sample_every),
            f"0x{row.addr_min:x}-0x{row.addr_max:x}",
        ]
        for row in rows
    ]
    svg = render_panels_svg(
        "Sampling Memory Trace Metrics",
        [
            f"matched_launches={meta.get('matched_launches', 0)} unique_kernels={meta.get('unique_kernels', len(rows))}",
            "Rows aggregate sampled memory-trace output into per-kernel totals and address spans.",
        ],
        [("Sampling Metrics", row_labels, columns, numeric, display)],
    )
    return csv_rows, svg


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    text = input_path.read_text(encoding="utf-8")
    tool = detect_tool(text) if args.tool == "auto" else args.tool

    parsers = {
        "gemm_wavefit": parse_wavefit,
        "gemm_orchestration": parse_orchestration,
        "epilogue_fusion": parse_epilogue,
        "tail_fragment": parse_tail,
        "reuse_distance": parse_reuse_distance,
        "pipeline_depth": parse_pipeline_depth,
        "tile_lifetime": parse_tile_lifetime,
        "bank_conflict": parse_bank_conflict,
        "register_pressure": parse_register_pressure,
        "cta_role": parse_cta_role,
        "kernel_summary": parse_kernel_summary,
        "sampling_mem_trace": parse_sampling,
    }
    builders = {
        "gemm_wavefit": build_wavefit_outputs,
        "gemm_orchestration": build_orchestration_outputs,
        "epilogue_fusion": build_epilogue_outputs,
        "tail_fragment": build_tail_outputs,
        "reuse_distance": build_reuse_distance_outputs,
        "pipeline_depth": build_pipeline_depth_outputs,
        "tile_lifetime": build_tile_lifetime_outputs,
        "bank_conflict": build_bank_conflict_outputs,
        "register_pressure": build_register_pressure_outputs,
        "cta_role": build_cta_role_outputs,
        "kernel_summary": build_kernel_summary_outputs,
        "sampling_mem_trace": build_sampling_outputs,
    }

    meta, rows = parsers[tool](text)
    if not rows:
        raise RuntimeError(f"no rows parsed from {input_path} for tool type {tool}")

    rows = sort_rows(tool, rows)[: max(1, args.top)]
    csv_rows, svg = builders[tool](meta, rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    csv_path = out_dir / f"{stem}.{tool}.csv"
    svg_path = out_dir / f"{stem}.{tool}.svg"
    write_csv(csv_path, csv_rows)
    svg_path.write_text(svg, encoding="utf-8")

    print(f"tool={tool}")
    print(f"rows={len(rows)}")
    print(f"csv={csv_path}")
    print(f"svg={svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
