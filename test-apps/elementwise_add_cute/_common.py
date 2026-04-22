#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time


def import_deps():
    try:
        import torch
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "These scripts require PyTorch and the CuTe DSL Python package.\n"
            "Install CUTLASS Python DSL support and run again.\n"
            f"Import error: {exc}"
        ) from exc
    return torch, cutlass, cute, from_dlpack


torch, cutlass, cute, from_dlpack = import_deps()


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--dtype", choices=["fp16"], default="fp16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    return parser.parse_args()


def torch_dtype_from_name(name: str):
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def make_inputs(args: argparse.Namespace):
    dtype = torch_dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    a_cpu = torch.randn(args.m, args.n, device="cpu", dtype=torch.float32)
    b_cpu = torch.randn(args.m, args.n, device="cpu", dtype=torch.float32)
    a = a_cpu.to(device="cuda", dtype=dtype)
    b = b_cpu.to(device="cuda", dtype=dtype)
    c = torch.zeros(args.m, args.n, device="cuda", dtype=dtype)

    return {
        "a_cpu": a_cpu,
        "b_cpu": b_cpu,
        "a": a,
        "b": b,
        "c": c,
        "a_view": from_dlpack(a, assumed_align=16),
        "b_view": from_dlpack(b, assumed_align=16),
        "c_view": from_dlpack(c, assumed_align=16),
    }


def benchmark(fn, call_args: tuple, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(*call_args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*call_args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


def finalize_report(
    *,
    method: str,
    args: argparse.Namespace,
    inputs: dict,
    avg_ms: float,
) -> None:
    y_cpu = inputs["c"].float().cpu()
    ref_cpu = (inputs["a_cpu"] + inputs["b_cpu"]).float()
    max_abs_err = (y_cpu - ref_cpu).abs().max().item()

    print(f"method={method} m={args.m} n={args.n} dtype={args.dtype}")
    print(f"avg_kernel_time_ms={avg_ms:.3f}")
    print(f"max_abs_err={max_abs_err:.6e}")
    print("device:", torch.cuda.get_device_name(0))
    print("cc:", torch.cuda.get_device_capability(0))
    print("sample_out:", float(inputs["c"].flatten()[0].float().item()))

