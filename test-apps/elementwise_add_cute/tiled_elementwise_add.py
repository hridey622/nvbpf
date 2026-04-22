#!/usr/bin/env python3
"""
Standalone CuTe DSL tiled elementwise-add kernel from elementwise_add.ipynb.

This is the TV-layout based version from the notebook. For the default fp16
path, keep shapes aligned to the 64x512 tile.

Example:
  ACK_CTX_INIT_LIMITATION=1 \
  NVBPF_KERNEL_FILTER=kernel_cutlass \
  LD_PRELOAD=$PWD/tools/nvbpf_examples/kernel_summary.so \
  python3 test-apps/elementwise_add_cute/tiled_elementwise_add.py
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))

from _common import benchmark, cute, finalize_report, make_inputs, parse_args, torch


@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tv_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]

    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    thr_coord = (tidx, None)
    thrA = tidfrgA[thr_coord]
    thrB = tidfrgB[thr_coord]
    thrC = tidfrgC[thr_coord]
    thrC[None] = thrA.load() + thrB.load()


@cute.jit
def tiled_elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    coalesced_ldst_bytes = 16
    dtype = mA.element_type

    thr_layout = cute.make_ordered_layout((4, 64), order=(1, 0))
    val_layout = cute.make_ordered_layout((16, coalesced_ldst_bytes), order=(1, 0))
    val_layout = cute.recast_layout(dtype.width, 8, val_layout)
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)

    remap_block = cute.make_ordered_layout(
        cute.select(gA.shape[1], mode=[1, 0]), order=(1, 0)
    )
    gA = cute.composition(gA, (None, remap_block))
    gB = cute.composition(gB, (None, remap_block))
    gC = cute.composition(gC, (None, remap_block))

    elementwise_add_kernel(gA, gB, gC, tv_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def main() -> int:
    args = parse_args("Run the tiled CuTe DSL elementwise-add kernel.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this script.")
    if args.m % 64 != 0 or args.n % 512 != 0:
        raise SystemExit("--m must be divisible by 64 and --n by 512 for this tiled fp16 layout.")

    inputs = make_inputs(args)
    fn = cute.compile(
        tiled_elementwise_add,
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
    finalize_report(method="tiled", args=args, inputs=inputs, avg_ms=avg_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

