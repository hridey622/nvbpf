#!/usr/bin/env python3
"""
Standalone CuTe DSL naive elementwise-add kernel from elementwise_add.ipynb.

Example:
  ACK_CTX_INIT_LIMITATION=1 \
  NVBPF_KERNEL_FILTER=kernel_cutlass \
  LD_PRELOAD=$PWD/tools/nvbpf_examples/kernel_summary.so \
  python3 test-apps/elementwise_add_cute/naive_elementwise_add.py
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))

from _common import benchmark, cute, finalize_report, make_inputs, parse_args, torch


@cute.kernel
def naive_elementwise_add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx
    m, n = gA.shape
    total = m * n
    if thread_idx < total:
        ni = thread_idx % n
        mi = thread_idx // n
        gC[mi, ni] = gA[mi, ni] + gB[mi, ni]


@cute.jit
def naive_elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    threads_per_block = 256
    m, n = mA.shape
    blocks = (m * n + threads_per_block - 1) // threads_per_block
    naive_elementwise_add_kernel(mA, mB, mC).launch(
        grid=(blocks, 1, 1),
        block=(threads_per_block, 1, 1),
    )


def main() -> int:
    args = parse_args("Run the naive CuTe DSL elementwise-add kernel.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this script.")

    inputs = make_inputs(args)
    fn = cute.compile(naive_elementwise_add, inputs["a_view"], inputs["b_view"], inputs["c_view"])
    avg_ms = benchmark(
        fn,
        (inputs["a_view"], inputs["b_view"], inputs["c_view"]),
        args.warmup,
        args.iters,
    )
    finalize_report(method="naive", args=args, inputs=inputs, avg_ms=avg_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

