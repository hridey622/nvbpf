# CuTe Elementwise Add Runners

These scripts split the four main CuTe DSL implementations from
[elementwise_add.ipynb](/home/hridey/nvbpf/elementwise_add.ipynb) into
standalone Python files so they can be instrumented with NV-BPF.

Files:

- `naive_elementwise_add.py`
- `vectorized_elementwise_add.py`
- `tiled_elementwise_add.py`
- `generic_elementwise_add.py`

## Recommended Instrumentation

Start with `kernel_summary.so` and filter to the generated CuTe kernels:

```bash
cd /home/hridey/nvbpf
ACK_CTX_INIT_LIMITATION=1 \
NVBPF_KERNEL_FILTER=kernel_cutlass \
LD_PRELOAD=$(pwd)/tools/nvbpf_examples/kernel_summary.so \
python3 test-apps/elementwise_add_cute/naive_elementwise_add.py
```

Swap in the other methods:

```bash
python3 test-apps/elementwise_add_cute/vectorized_elementwise_add.py
python3 test-apps/elementwise_add_cute/tiled_elementwise_add.py
python3 test-apps/elementwise_add_cute/generic_elementwise_add.py
```

For sampled memory behavior:

```bash
ACK_CTX_INIT_LIMITATION=1 \
NVBPF_KERNEL_FILTER=kernel_cutlass \
NVBPF_SAMPLE_EVERY=64 \
LD_PRELOAD=$(pwd)/tools/nvbpf_examples/sampling_mem_trace.so \
python3 test-apps/elementwise_add_cute/vectorized_elementwise_add.py
```

For branch behavior:

```bash
ACK_CTX_INIT_LIMITATION=1 \
NVBPF_KERNEL_FILTER=kernel_cutlass \
NVBPF_MIN_ACTIVE_LANES=8 \
LD_PRELOAD=$(pwd)/tools/nvbpf_examples/branch_divergence.so \
python3 test-apps/elementwise_add_cute/generic_elementwise_add.py
```

## Shape Notes

- `vectorized_elementwise_add.py` expects `--n` divisible by `8`
- `tiled_elementwise_add.py` expects fp16 shapes aligned to `64 x 512`
- `generic_elementwise_add.py` currently uses the same fp16 tile assumptions

Defaults are chosen to satisfy those constraints.
