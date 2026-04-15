#!/usr/bin/env python3
"""
CuTe DSL AXPY+ReLU driver for NV-BPF / NVBit instrumentation.

This compiles a small CuTe DSL kernel and launches it on CUDA so it can be
observed with LD_PRELOAD'ed NV-BPF tools such as kernel_summary.so.

Example:
  ACK_CTX_INIT_LIMITATION=1 \
  LD_PRELOAD=$PWD/tools/nvbpf_examples/kernel_summary.so \
  python3 test-apps/axpy_relu_cute/axpy_relu_cute.py
"""

from __future__ import annotations

import argparse
import sys
import time


def _import_deps():
    try:
        import torch
        import cutlass
        import cutlass.cute as cute
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "This script requires PyTorch and CuTe DSL Python.\n"
            "Install the CUTLASS Python DSL package and run again.\n"
            f"Import error: {exc}"
        ) from exc
    return torch, cutlass, cute


torch, cutlass, cute = _import_deps()


@cute.kernel
def device_axpy_relu(
    a: cute.Tensor,
    b: cute.Tensor,
    y: cute.Tensor,
    alpha: cutlass.Float32,
):
    threads_per_block = 256

    bx, _, _ = cute.arch.block_idx()
    tx, _, _ = cute.arch.thread_idx()

    tid = bx * threads_per_block + tx
    n = a.shape[0]
    if tid < n:
        val = a[tid] + b[tid] * alpha
        y[tid] = val if val > 0.0 else 0.0


@cute.jit
def axpy_relu(
    a: cute.Tensor,
    b: cute.Tensor,
    y: cute.Tensor,
    alpha: cutlass.Float32,
):
    n = a.shape[0]
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    device_axpy_relu(a, b, y, alpha).launch(
        grid=(blocks, 1, 1),
        block=(threads_per_block, 1, 1),
    )


def build_axpy_relu():
    n = cute.sym_int()
    a_fake = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
    b_fake = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
    y_fake = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))

    return cute.compile(
        axpy_relu,
        a_fake,
        b_fake,
        y_fake,
        cutlass.Float32(1.0),
        options="--enable-tvm-ffi",
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CuTe DSL AXPY+ReLU kernel.")
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    return parser.parse_args(argv)


def make_inputs(n: int, seed: int):
    torch.manual_seed(seed)
    a_cpu = torch.randn(n, device="cpu", dtype=torch.float32)
    b_cpu = torch.randn(n, device="cpu", dtype=torch.float32)
    a = a_cpu.to(device="cuda", dtype=torch.float32)
    b = b_cpu.to(device="cuda", dtype=torch.float32)
    y = torch.empty_like(a)
    return a, b, y


def run(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this example.")

    fn = build_axpy_relu()
    a, b, y = make_inputs(args.n, args.seed)

    for _ in range(args.warmup):
        fn(a, b, y, args.alpha)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.iters):
        fn(a, b, y, args.alpha)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1e3 / args.iters

    ref = torch.relu(a + args.alpha * b)
    max_abs_err = (y - ref).abs().max().item()

    print(f"n={args.n} alpha={args.alpha}")
    print(f"avg_kernel_time_ms={elapsed_ms:.3f}")
    print(f"max_abs_err={max_abs_err:.6e}")
    print("device:", torch.cuda.get_device_name(0))
    print("cc:", torch.cuda.get_device_capability(0))
    print("sample_out:", float(y[0].item()))
    return 0


def main(argv: list[str]) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
