from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CounterSpec:
    name: str
    opcodes: tuple[str, ...] = ()
    loads: bool = False
    stores: bool = False
    branches: bool = False
    exclude_constant_loads: bool = True
    description: str = ""


@dataclass(frozen=True)
class EventFieldSpec:
    name: str
    type_name: str


@dataclass(frozen=True)
class EventSpec:
    name: str
    fields: tuple[EventFieldSpec, ...] = ()
    capacity: int = 8192
    description: str = ""


@dataclass(frozen=True)
class ApiTraceSpec:
    name: str
    callbacks: tuple[str, ...] = ()
    on_exit: bool = True
    correlate_launches: bool = False
    description: str = ""


@dataclass(frozen=True)
class DeviceHookSpec:
    name: str
    args: tuple[str, ...]
    source: str
    opcodes: tuple[str, ...] = ()
    loads: bool = False
    stores: bool = False
    branches: bool = False
    exclude_constant_loads: bool = True
    description: str = ""


@dataclass(frozen=True)
class MapSpec:
    name: str
    kind: str
    type_name: str
    length: int
    description: str = ""


@dataclass(frozen=True)
class GemmWavefitSpec:
    sm_cta_entries_map: str
    active_sm_bitmap_map: str
    filter_csv: str = (
        "gemm,sgemm,hgemm,dgemm,bgemm,igemm,matmul,cublas,cutlass"
    )
    filter_env: str = "NVBPF_GEMM_FILTER"


@dataclass(frozen=True)
class GemmOrchestrationSpec:
    filter_csv: str = (
        "gemm,sgemm,hgemm,dgemm,bgemm,igemm,matmul,cublas,cutlass"
    )
    filter_env: str = "NVBPF_GEMM_FILTER"
    window_env: str = "NVBPF_ORCH_WINDOW"
    default_window: int = 2


@dataclass(frozen=True)
class ToolSpec:
    name: str
    maps: tuple[MapSpec, ...] = field(default_factory=tuple)
    counters: tuple[CounterSpec, ...] = field(default_factory=tuple)
    events: tuple[EventSpec, ...] = field(default_factory=tuple)
    device_hooks: tuple[DeviceHookSpec, ...] = field(default_factory=tuple)
    api_traces: tuple[ApiTraceSpec, ...] = field(default_factory=tuple)
    gemm_wavefit: GemmWavefitSpec | None = None
    gemm_orchestration: GemmOrchestrationSpec | None = None
    kernel_filter_env: str = "NVBPF_KERNEL_FILTER"
    banner: str = ""
