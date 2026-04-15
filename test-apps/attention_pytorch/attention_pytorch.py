#!/usr/bin/env python3
"""
PyTorch attention driver for NV-BPF / NVBit instrumentation.

This script exercises torch.nn.functional.scaled_dot_product_attention so the
Python process launches real CUDA attention kernels that can be observed with
LD_PRELOAD'ed NV-BPF tools such as attention_debug.so.

Example:
  ACK_CTX_INIT_LIMITATION=1 \
  LD_PRELOAD=$PWD/tools/nvbpf_examples/attention_debug.so \
  python3 test-apps/attention_pytorch/attention_pytorch.py \
      --batch 1 --heads 4 --seq-len 128 --head-dim 64 --dtype fp16
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
import time

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
except Exception:  # pragma: no cover - version dependent
    sdpa_kernel = None
    SDPBackend = None


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PyTorch scaled dot-product attention on CUDA."
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--backend", choices=["auto", "math", "flash", "mem_efficient"], default="math")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    return parser.parse_args(argv)


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def sdpa_backend_summary() -> str:
    parts: list[str] = []
    try:
        parts.append(f"flash_enabled={torch.backends.cuda.flash_sdp_enabled()}")
        parts.append(f"mem_efficient_enabled={torch.backends.cuda.mem_efficient_sdp_enabled()}")
        parts.append(f"math_enabled={torch.backends.cuda.math_sdp_enabled()}")
    except Exception as exc:  # pragma: no cover - version dependent
        parts.append(f"backend_query_error={exc}")
    return " ".join(parts)


def sdpa_context(backend_name: str):
    if backend_name == "auto" or sdpa_kernel is None or SDPBackend is None:
        return contextlib.nullcontext()

    backend_map = {
        "math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
    }
    return sdpa_kernel(backends=[backend_map[backend_name]])


def make_inputs(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    shape = (batch, heads, seq_len, head_dim)
    q = torch.randn(shape, device="cpu", dtype=torch.float32).to(device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cpu", dtype=torch.float32).to(device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cpu", dtype=torch.float32).to(device="cuda", dtype=dtype)
    return q, k, v


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q.float(), k.transpose(-1, -2).float()) * scale
    if causal:
        s = q.size(-2)
        mask = torch.triu(torch.ones((s, s), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float())


def run(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this script.")

    dtype = torch_dtype_from_name(args.dtype)
    q, k, v = make_inputs(
        args.batch, args.heads, args.seq_len, args.head_dim, dtype, args.seed
    )

    with sdpa_context(args.backend):
        for _ in range(args.warmup):
            _ = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=args.dropout,
                is_causal=args.causal,
            )
        torch.cuda.synchronize()

        start = time.perf_counter()
        out = None
        for _ in range(args.iters):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=args.dropout,
                is_causal=args.causal,
            )
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1e3 / args.iters

    assert out is not None
    ref = reference_attention(q, k, v, args.causal)
    max_abs_err = (out.float() - ref).abs().max().item()

    print(
        f"batch={args.batch} heads={args.heads} seq_len={args.seq_len} "
        f"head_dim={args.head_dim} dtype={args.dtype} causal={args.causal} "
        f"backend={args.backend}"
    )
    print(f"avg_kernel_time_ms={elapsed_ms:.3f}")
    print(f"max_abs_err={max_abs_err:.6e}")
    print(sdpa_backend_summary())
    print("sample_out=", float(out[0, 0, 0, 0].float().item()))

    return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
