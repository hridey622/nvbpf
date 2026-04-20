from __future__ import annotations

import inspect
import textwrap
from typing import Any

from .model import (
    ApiTraceSpec,
    CounterSpec,
    DeviceHookSpec,
    EventFieldSpec,
    EventSpec,
    GemmOrchestrationSpec,
    GemmWavefitSpec,
    MapSpec,
    ToolSpec,
)


class _CounterField:
    def __init__(
        self,
        *,
        opcodes: list[str] | tuple[str, ...] | None = None,
        loads: bool = False,
        stores: bool = False,
        branches: bool = False,
        exclude_constant_loads: bool = True,
        description: str = "",
    ) -> None:
        self.opcodes = tuple(opcodes or ())
        self.loads = loads
        self.stores = stores
        self.branches = branches
        self.exclude_constant_loads = exclude_constant_loads
        self.description = description

    def to_spec(self, name: str) -> CounterSpec:
        return CounterSpec(
            name=name,
            opcodes=self.opcodes,
            loads=self.loads,
            stores=self.stores,
            branches=self.branches,
            exclude_constant_loads=self.exclude_constant_loads,
            description=self.description,
        )


class _EventField:
    def __init__(
        self,
        *,
        fields: dict[str, str],
        capacity: int = 8192,
        description: str = "",
    ) -> None:
        if not fields:
            raise ValueError("event() requires at least one field")
        self.fields = tuple(EventFieldSpec(name=k, type_name=v) for k, v in fields.items())
        self.capacity = capacity
        self.description = description

    def to_spec(self, name: str) -> EventSpec:
        return EventSpec(
            name=name,
            fields=self.fields,
            capacity=self.capacity,
            description=self.description,
        )


class _ApiTraceField:
    def __init__(
        self,
        *,
        callbacks: list[str] | tuple[str, ...],
        on_exit: bool = True,
        correlate_launches: bool = False,
        description: str = "",
    ) -> None:
        if not callbacks:
            raise ValueError("api_trace() requires at least one callback")
        self.callbacks = tuple(callbacks)
        self.on_exit = on_exit
        self.correlate_launches = correlate_launches
        self.description = description

    def to_spec(self, name: str) -> ApiTraceSpec:
        return ApiTraceSpec(
            name=name,
            callbacks=self.callbacks,
            on_exit=self.on_exit,
            correlate_launches=self.correlate_launches,
            description=self.description,
        )


class _MapField:
    def __init__(
        self,
        *,
        kind: str,
        type_name: str,
        length: int,
        description: str = "",
    ) -> None:
        if length <= 0:
            raise ValueError("map length must be positive")
        self.kind = kind
        self.type_name = type_name
        self.length = length
        self.description = description

    def to_spec(self, name: str) -> MapSpec:
        return MapSpec(
            name=name,
            kind=self.kind,
            type_name=self.type_name,
            length=self.length,
            description=self.description,
        )


class _GemmWavefitField:
    def __init__(
        self,
        *,
        sm_cta_entries_map: str,
        active_sm_bitmap_map: str,
        filter_csv: str = (
            "gemm,sgemm,hgemm,dgemm,bgemm,igemm,matmul,cublas,cutlass"
        ),
        filter_env: str = "NVBPF_GEMM_FILTER",
    ) -> None:
        self.sm_cta_entries_map = sm_cta_entries_map
        self.active_sm_bitmap_map = active_sm_bitmap_map
        self.filter_csv = filter_csv
        self.filter_env = filter_env

    def to_spec(self) -> GemmWavefitSpec:
        return GemmWavefitSpec(
            sm_cta_entries_map=self.sm_cta_entries_map,
            active_sm_bitmap_map=self.active_sm_bitmap_map,
            filter_csv=self.filter_csv,
            filter_env=self.filter_env,
        )


class _GemmOrchestrationField:
    def __init__(
        self,
        *,
        filter_csv: str = (
            "gemm,sgemm,hgemm,dgemm,bgemm,igemm,matmul,cublas,cutlass"
        ),
        filter_env: str = "NVBPF_GEMM_FILTER",
        window_env: str = "NVBPF_ORCH_WINDOW",
        default_window: int = 2,
    ) -> None:
        self.filter_csv = filter_csv
        self.filter_env = filter_env
        self.window_env = window_env
        self.default_window = default_window

    def to_spec(self) -> GemmOrchestrationSpec:
        return GemmOrchestrationSpec(
            filter_csv=self.filter_csv,
            filter_env=self.filter_env,
            window_env=self.window_env,
            default_window=self.default_window,
        )


class _DeviceHookDecorator:
    def __init__(
        self,
        *,
        opcodes: list[str] | tuple[str, ...] | None = None,
        loads: bool = False,
        stores: bool = False,
        branches: bool = False,
        exclude_constant_loads: bool = True,
        description: str = "",
    ) -> None:
        self.opcodes = tuple(opcodes or ())
        self.loads = loads
        self.stores = stores
        self.branches = branches
        self.exclude_constant_loads = exclude_constant_loads
        self.description = description

    def __call__(self, fn: Any) -> Any:
        if not self.opcodes and not self.loads and not self.stores and not self.branches:
            raise ValueError("device_hook() requires at least one trigger predicate")
        source = textwrap.dedent(inspect.getsource(fn))
        sig = inspect.signature(fn)
        fn._nvbpf_device_hook = DeviceHookSpec(
            name=fn.__name__,
            args=tuple(sig.parameters.keys()),
            source=source,
            opcodes=self.opcodes,
            loads=self.loads,
            stores=self.stores,
            branches=self.branches,
            exclude_constant_loads=self.exclude_constant_loads,
            description=self.description,
        )
        return fn


def counter(
    *,
    opcodes: list[str] | tuple[str, ...] | None = None,
    loads: bool = False,
    stores: bool = False,
    branches: bool = False,
    exclude_constant_loads: bool = True,
    description: str = "",
) -> _CounterField:
    if not opcodes and not loads and not stores and not branches:
        raise ValueError("counter() requires at least one of opcodes/loads/stores/branches")
    return _CounterField(
        opcodes=opcodes,
        loads=loads,
        stores=stores,
        branches=branches,
        exclude_constant_loads=exclude_constant_loads,
        description=description,
    )


def event(
    *,
    fields: dict[str, str],
    capacity: int = 8192,
    description: str = "",
) -> _EventField:
    return _EventField(fields=fields, capacity=capacity, description=description)


def api_trace(
    *,
    callbacks: list[str] | tuple[str, ...],
    on_exit: bool = True,
    correlate_launches: bool = False,
    description: str = "",
) -> _ApiTraceField:
    return _ApiTraceField(
        callbacks=callbacks,
        on_exit=on_exit,
        correlate_launches=correlate_launches,
        description=description,
    )


def device_hook(
    *,
    opcodes: list[str] | tuple[str, ...] | None = None,
    loads: bool = False,
    stores: bool = False,
    branches: bool = False,
    exclude_constant_loads: bool = True,
    description: str = "",
) -> _DeviceHookDecorator:
    return _DeviceHookDecorator(
        opcodes=opcodes,
        loads=loads,
        stores=stores,
        branches=branches,
        exclude_constant_loads=exclude_constant_loads,
        description=description,
    )


def array(
    *,
    type_name: str,
    length: int,
    description: str = "",
) -> _MapField:
    return _MapField(
        kind="array",
        type_name=type_name,
        length=length,
        description=description,
    )


def percpu_array(
    *,
    type_name: str,
    length: int,
    description: str = "",
) -> _MapField:
    return _MapField(
        kind="percpu_array",
        type_name=type_name,
        length=length,
        description=description,
    )


def gemm_wavefit(
    *,
    sm_cta_entries_map: str,
    active_sm_bitmap_map: str,
    filter_csv: str = (
        "gemm,sgemm,hgemm,dgemm,bgemm,igemm,matmul,cublas,cutlass"
    ),
    filter_env: str = "NVBPF_GEMM_FILTER",
) -> _GemmWavefitField:
    return _GemmWavefitField(
        sm_cta_entries_map=sm_cta_entries_map,
        active_sm_bitmap_map=active_sm_bitmap_map,
        filter_csv=filter_csv,
        filter_env=filter_env,
    )


def gemm_orchestration_map(
    *,
    filter_csv: str = (
        "gemm,sgemm,hgemm,dgemm,bgemm,igemm,matmul,cublas,cutlass"
    ),
    filter_env: str = "NVBPF_GEMM_FILTER",
    window_env: str = "NVBPF_ORCH_WINDOW",
    default_window: int = 2,
) -> _GemmOrchestrationField:
    return _GemmOrchestrationField(
        filter_csv=filter_csv,
        filter_env=filter_env,
        window_env=window_env,
        default_window=default_window,
    )


def count(name: str) -> None:
    raise RuntimeError("count() is only valid inside an NV-BPF Python hook body")


def emit(event_name: str, **kwargs: Any) -> None:
    raise RuntimeError("emit() is only valid inside an NV-BPF Python hook body")


def tool(
    name: str,
    *,
    kernel_filter_env: str = "NVBPF_KERNEL_FILTER",
    banner: str | None = None,
):
    def decorator(cls: type[Any]) -> type[Any]:
        maps: list[MapSpec] = []
        counters: list[CounterSpec] = []
        events: list[EventSpec] = []
        api_traces: list[ApiTraceSpec] = []
        device_hooks: list[DeviceHookSpec] = []
        gemm_wavefit_spec: GemmWavefitSpec | None = None
        gemm_orchestration_spec: GemmOrchestrationSpec | None = None

        for attr_name, value in cls.__dict__.items():
            if isinstance(value, _MapField):
                maps.append(value.to_spec(attr_name))
            elif isinstance(value, _CounterField):
                counters.append(value.to_spec(attr_name))
            elif isinstance(value, _EventField):
                events.append(value.to_spec(attr_name))
            elif isinstance(value, _ApiTraceField):
                api_traces.append(value.to_spec(attr_name))
            elif isinstance(value, _GemmWavefitField):
                if gemm_wavefit_spec is not None:
                    raise ValueError(f"tool {name!r} declares multiple gemm_wavefit() analyses")
                gemm_wavefit_spec = value.to_spec()
            elif isinstance(value, _GemmOrchestrationField):
                if gemm_orchestration_spec is not None:
                    raise ValueError(
                        f"tool {name!r} declares multiple gemm_orchestration_map() analyses"
                    )
                gemm_orchestration_spec = value.to_spec()

            hook_spec = getattr(value, "_nvbpf_device_hook", None)
            if isinstance(hook_spec, DeviceHookSpec):
                device_hooks.append(hook_spec)

        if gemm_wavefit_spec is not None and gemm_orchestration_spec is not None:
            raise ValueError(
                f"tool {name!r} cannot combine gemm_wavefit() and gemm_orchestration_map()"
            )

        maps_by_name = {spec.name: spec for spec in maps}
        if gemm_wavefit_spec is not None:
            if counters or events or api_traces or device_hooks:
                raise ValueError(
                    f"tool {name!r} uses gemm_wavefit(); combine it only with explicit maps in this DSL version"
                )
            sm_entries = maps_by_name.get(gemm_wavefit_spec.sm_cta_entries_map)
            if sm_entries is None:
                raise ValueError(
                    f"gemm_wavefit() references unknown map {gemm_wavefit_spec.sm_cta_entries_map!r}"
                )
            if sm_entries.kind != "percpu_array" or sm_entries.type_name != "u64":
                raise ValueError(
                    "gemm_wavefit() requires sm_cta_entries_map to be a percpu_array(type_name='u64', ...)"
                )
            bitmap = maps_by_name.get(gemm_wavefit_spec.active_sm_bitmap_map)
            if bitmap is None:
                raise ValueError(
                    f"gemm_wavefit() references unknown map {gemm_wavefit_spec.active_sm_bitmap_map!r}"
                )
            if bitmap.kind != "array" or bitmap.type_name != "u64":
                raise ValueError(
                    "gemm_wavefit() requires active_sm_bitmap_map to be an array(type_name='u64', ...)"
                )
        if gemm_orchestration_spec is not None and (maps or counters or events or api_traces or device_hooks):
            raise ValueError(
                f"tool {name!r} uses gemm_orchestration_map(); keep it host-only in this DSL version"
            )

        if (
            not maps
            and not counters
            and not events
            and not api_traces
            and not device_hooks
            and gemm_wavefit_spec is None
            and gemm_orchestration_spec is None
        ):
            raise ValueError(
                f"tool {name!r} has no maps, counters, events, hooks, api traces, or analyses"
            )

        cls.tool_spec = ToolSpec(
            name=name,
            maps=tuple(maps),
            counters=tuple(counters),
            events=tuple(events),
            device_hooks=tuple(device_hooks),
            api_traces=tuple(api_traces),
            gemm_wavefit=gemm_wavefit_spec,
            gemm_orchestration=gemm_orchestration_spec,
            kernel_filter_env=kernel_filter_env,
            banner=banner or name.upper(),
        )
        return cls

    return decorator
