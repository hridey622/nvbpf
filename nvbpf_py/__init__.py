"""Python-first DSL for generating NV-BPF tools."""

from .dsl import (
    api_trace,
    array,
    count,
    counter,
    device_hook,
    emit,
    event,
    gemm_orchestration_map,
    gemm_wavefit,
    percpu_array,
    tool,
)
from .model import (
    ApiTraceSpec,
    CounterSpec,
    DeviceHookSpec,
    EventSpec,
    GemmOrchestrationSpec,
    GemmWavefitSpec,
    MapSpec,
    ToolSpec,
)

__all__ = [
    "api_trace",
    "array",
    "count",
    "counter",
    "device_hook",
    "emit",
    "event",
    "gemm_orchestration_map",
    "gemm_wavefit",
    "percpu_array",
    "tool",
    "ApiTraceSpec",
    "CounterSpec",
    "DeviceHookSpec",
    "EventSpec",
    "GemmOrchestrationSpec",
    "GemmWavefitSpec",
    "MapSpec",
    "ToolSpec",
]
