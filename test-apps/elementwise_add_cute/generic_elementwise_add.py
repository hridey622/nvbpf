#!/usr/bin/env python3
"""
Standalone CuTe DSL generic elementwise-apply(add) kernel from elementwise_add.ipynb.

This keeps the notebook's generic list-of-inputs formulation, but hardcodes
addition inside the kernel because this local CuTe runtime does not accept
Python callables as `Constexpr` JIT arguments.

Example:
  ACK_CTX_INIT_LIMITATION=1 \
  NVBPF_KERNEL_FILTER=kernel_cutlass \
  LD_PRELOAD=$PWD/tools/nvbpf_examples/kernel_summary.so \
  python3 test-apps/elementwise_add_cute/generic_elementwise_add.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))

from _common import benchmark, cutlass, cute, finalize_report, make_inputs, parse_args, torch


@cute.kernel
def elementwise_apply_kernel(
    mInputs: List[cute.Tensor],
    mC: cute.Tensor,
    cC: cute.Tensor,
    shape: cute.Shape,
    tv_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_crd = ((None, None), bidx)
    gInputs = [t[blk_crd] for t in mInputs]
    gC = mC[blk_crd]
    gCrd = cC[blk_crd]

    tidfrgInputs = [cute.composition(t, tv_layout) for t in gInputs]
    tidfrgC = cute.composition(gC, tv_layout)
    tidfrgCrd = cute.composition(gCrd, tv_layout)

    thr_crd = (tidx, cute.repeat_like(None, tidfrgInputs[0][1]))
    thrInputs = [t[thr_crd] for t in tidfrgInputs]
    thrC = tidfrgC[thr_crd]
    thrCrd = tidfrgCrd[thr_crd]

    frgPred = cute.make_fragment(thrCrd.shape, cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(frgPred)):
        frgPred[i] = cute.elem_less(thrCrd[i], shape)

    loaded = [thrInput.load() for thrInput in thrInputs]
    result = loaded[0] + loaded[1]
    thrC.store(result)


@cute.jit
def generic_elementwise_add(inputs, result: cute.Tensor):
    coalesced_ldst_bytes = 16
    dtype = inputs[0].element_type

    thr_layout = cute.make_ordered_layout((4, 64), order=(1, 0))
    val_layout = cute.make_ordered_layout((16, coalesced_ldst_bytes), order=(1, 0))
    val_layout = cute.recast_layout(dtype.width, 8, val_layout)
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    mInputs = [cute.zipped_divide(t, tiler_mn) for t in inputs]
    mC = cute.zipped_divide(result, tiler_mn)

    remap_block = cute.make_ordered_layout(
        cute.select(mInputs[0].shape[1], mode=[1, 0]), order=(1, 0)
    )
    for i, tensor in enumerate(mInputs):
        mInputs[i] = cute.composition(tensor, (None, remap_block))
    mC = cute.composition(mC, (None, remap_block))

    idC = cute.make_identity_tensor(result.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)

    elementwise_apply_kernel(mInputs, mC, cC, result.shape, tv_layout).launch(
        grid=[cute.size(mC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def main() -> int:
    args = parse_args("Run the generic CuTe DSL elementwise-apply(add) kernel.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this script.")
    if args.m % 64 != 0 or args.n % 512 != 0:
        raise SystemExit("--m must be divisible by 64 and --n by 512 for this generic fp16 layout.")

    inputs = make_inputs(args)
    fn = cute.compile(generic_elementwise_add, [inputs["a_view"], inputs["b_view"]], inputs["c_view"])
    avg_ms = benchmark(
        fn,
        ([inputs["a_view"], inputs["b_view"]], inputs["c_view"]),
        args.warmup,
        args.iters,
    )
    finalize_report(method="generic_apply_add", args=args, inputs=inputs, avg_ms=avg_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
