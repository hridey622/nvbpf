#!/usr/bin/env python3
"""
Standalone CuTe DSL vectorized elementwise-add kernel from elementwise_add.ipynb.

Example:
  ACK_CTX_INIT_LIMITATION=1 \
  NVBPF_KERNEL_FILTER=kernel_cutlass \
  LD_PRELOAD=$PWD/tools/nvbpf_examples/kernel_summary.so \
  python3 test-apps/elementwise_add_cute/vectorized_elementwise_add.py
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))

from _common import benchmark, cute, finalize_report, make_inputs, parse_args, torch


@cute.kernel
def vectorized_elementwise_add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx
    m, n = gA.shape[1]
    total = m * n
    if thread_idx < total:
        ni = thread_idx % n
        mi = thread_idx // n
        gC[(None, (mi, ni))] = gA[(None, (mi, ni))].load() + gB[(None, (mi, ni))].load()


@cute.jit
def vectorized_elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    threads_per_block = 256
    gA = cute.zipped_divide(mA, (1, 8))
    gB = cute.zipped_divide(mB, (1, 8))
    gC = cute.zipped_divide(mC, (1, 8))
    grid_x = (cute.size(gC, mode=[1]) + threads_per_block - 1) // threads_per_block
    vectorized_elementwise_add_kernel(gA, gB, gC).launch(
        grid=(grid_x, 1, 1),
        block=(threads_per_block, 1, 1),
    )


def main() -> int:
    args = parse_args("Run the vectorized CuTe DSL elementwise-add kernel.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this script.")
    if args.n % 8 != 0:
        raise SystemExit("--n must be divisible by 8 for the vectorized layout.")

    inputs = make_inputs(args)
    fn = cute.compile(
        vectorized_elementwise_add,
        inputs["a_view"],
        inputs["b_view"],
        inputs["c_view"],
    )
    avg_ms = benchmark(
        fn,
        (inputs["a_view"], inputs["b_view"], inputs["c_view"]),
        args.warmup,
        args.iters,
    )
    finalize_report(method="vectorized", args=args, inputs=inputs, avg_ms=avg_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

