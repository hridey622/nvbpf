#!/usr/bin/env python3
"""
PyTorch Conv2d driver for NV-BPF / NVBit instrumentation.

This keeps tensor initialization on CPU first, then moves data to CUDA, so the
instrumented run focuses more on convolution kernels than on setup noise.

Example:
  ACK_CTX_INIT_LIMITATION=1 \
  NVBPF_KERNEL_FILTER=conv \
  LD_PRELOAD=$PWD/tools/nvbpf_examples/kernel_summary.so \
  python3 test-apps/conv_pytorch/conv2d_pytorch.py
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn as nn


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CUDA Conv2d workload.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--out-channels", type=int, default=16)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--padding", type=int, default=1)
    parser.add_argument("--bias", action="store_true", default=True)
    parser.add_argument("--no-bias", dest="bias", action="store_false")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    return parser.parse_args(argv)


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def make_input(args: argparse.Namespace, dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(args.seed)
    x_cpu = torch.randn(
        args.batch,
        args.in_channels,
        args.height,
        args.width,
        device="cpu",
        dtype=torch.float32,
    )
    return x_cpu.to(device="cuda", dtype=dtype)


def run(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this script.")

    dtype = torch_dtype_from_name(args.dtype)
    x = make_input(args, dtype)

    conv = nn.Conv2d(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        bias=args.bias,
    ).to(device="cuda", dtype=dtype)

    for _ in range(args.warmup):
        _ = conv(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    y = None
    for _ in range(args.iters):
        y = conv(x)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1e3 / args.iters

    assert y is not None
    print(
        f"batch={args.batch} in_channels={args.in_channels} "
        f"out_channels={args.out_channels} height={args.height} width={args.width} "
        f"kernel={args.kernel_size} stride={args.stride} padding={args.padding} "
        f"dtype={args.dtype} bias={args.bias}"
    )
    print(f"avg_kernel_time_ms={elapsed_ms:.3f}")
    print("input shape :", tuple(x.shape))
    print("output shape:", tuple(y.shape))
    print("device      :", y.device)
    print(f"cudnn_enabled={torch.backends.cudnn.enabled}")
    print(f"tf32_matmul_allowed={torch.backends.cuda.matmul.allow_tf32}")
    print(f"tf32_cudnn_allowed={torch.backends.cudnn.allow_tf32}")
    print("sample_out  :", float(y.flatten()[0].float().item()))
    return 0


def main(argv: list[str]) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
