#!/usr/bin/env python3
"""
Naive single-head attention in CuTe DSL Python.

This is intentionally simple and instrumentation-friendly rather than fast.
Each CTA handles one query row and thread 0 performs the whole row's work.
That makes it useful for validating that NVBit / NV-BPF can observe a CuTe DSL
generated kernel, even though it is not a production-quality attention kernel.

Example:
  ACK_CTX_INIT_LIMITATION=1 \
  LD_PRELOAD=$PWD/tools/nvbpf_examples/attention_debug.so \
  python3 test-apps/attention_cute/attention_cute.py --seq-len 32 --head-dim 64
"""

from __future__ import annotations

import argparse
import math
import sys
import time


def _import_deps():
    try:
        import torch
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute import arch
        from cutlass.cute.runtime import make_ptr
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "This script requires PyTorch and CuTe DSL Python.\n"
            "Install the CUTLASS Python DSL package and run again.\n"
            f"Import error: {exc}"
        ) from exc
    return torch, cutlass, cute, arch, make_ptr


torch, cutlass, cute, arch, make_ptr = _import_deps()


@cute.kernel
def attention_kernel(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    seq_len: cutlass.Int32,
    head_dim: cutlass.Int32,
    scale: cutlass.Float32,
):
    tidx, _, _ = arch.thread_idx()
    bidx, _, _ = arch.block_idx()

    row = bidx
    if row >= seq_len or tidx != 0:
        return

    minus_inf = cutlass.Float32(-1.0e20)
    max_score = minus_inf

    # Pass 1: compute the row max for numerically stable softmax.
    for col in range(seq_len):
        score = cutlass.Float32(0.0)
        q_base = (row * head_dim)
        k_base = (col * head_dim)
        for d in range(head_dim):
            q_val = arch.load(q_ptr + (q_base + d), cutlass.Float32)
            k_val = arch.load(k_ptr + (k_base + d), cutlass.Float32)
            score = score + q_val * k_val
        score = score * scale
        if score > max_score:
            max_score = score

    denom = cutlass.Float32(0.0)

    # Pass 2: accumulate exp(score - max) * V and the denominator.
    for d in range(head_dim):
        arch.store(o_ptr + (row * head_dim + d), cutlass.Float32(0.0))

    for col in range(seq_len):
        score = cutlass.Float32(0.0)
        q_base = (row * head_dim)
        k_base = (col * head_dim)
        v_base = (col * head_dim)
        for d in range(head_dim):
            q_val = arch.load(q_ptr + (q_base + d), cutlass.Float32)
            k_val = arch.load(k_ptr + (k_base + d), cutlass.Float32)
            score = score + q_val * k_val
        score = score * scale
        weight = cute.exp(score - max_score)
        denom = denom + weight

        for d in range(head_dim):
            out_idx = row * head_dim + d
            old_val = arch.load(o_ptr + out_idx, cutlass.Float32)
            v_val = arch.load(v_ptr + (v_base + d), cutlass.Float32)
            arch.store(o_ptr + out_idx, old_val + weight * v_val)

    # Pass 3: normalize.
    for d in range(head_dim):
        out_idx = row * head_dim + d
        old_val = arch.load(o_ptr + out_idx, cutlass.Float32)
        arch.store(o_ptr + out_idx, old_val / denom)


@cute.jit
def attention_wrapper(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    seq_len: cutlass.Int32,
    head_dim: cutlass.Int32,
    scale: cutlass.Float32,
):
    attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        seq_len,
        head_dim,
        scale,
    ).launch(grid=[seq_len, 1, 1], block=[32, 1, 1])


def torch_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def run_attention(seq_len: int, head_dim: int, dtype: str, seed: int) -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this example.")

    if dtype != "fp32":
        raise SystemExit("This example currently supports only --dtype fp32.")

    torch.manual_seed(seed)
    device = "cuda"
    q = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
    o = torch.empty_like(q)

    q_ptr = make_ptr(cutlass.Float32, q.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    k_ptr = make_ptr(cutlass.Float32, k.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    v_ptr = make_ptr(cutlass.Float32, v.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    o_ptr = make_ptr(cutlass.Float32, o.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    scale = 1.0 / math.sqrt(head_dim)

    start = time.perf_counter()
    attention_wrapper(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        seq_len,
        head_dim,
        scale,
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1e3

    ref = torch_attention_reference(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
    ).squeeze(0)
    max_abs_err = (o - ref).abs().max().item()

    print(f"seq_len={seq_len} head_dim={head_dim} dtype={dtype}")
    print(f"kernel_time_ms={elapsed_ms:.3f}")
    print(f"max_abs_err={max_abs_err:.6e}")
    print("sample_out_row0_col0=", float(o[0, 0].item()))

    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a naive CuTe DSL attention kernel.")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp32"], default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    return run_attention(args.seq_len, args.head_dim, args.dtype, args.seed)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
