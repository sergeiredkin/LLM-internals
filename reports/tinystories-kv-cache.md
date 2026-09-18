# TinyStories KV-Cache Generation Report

**Date:** 2026-09-18

## Question

Does reusing per-layer keys and values accelerate autoregressive generation for the 26M
RoPE + SwiGLU + GQA model without changing model weights or materially changing outputs?

## Implementation

Cached generation has two phases:

1. **Prefill:** process the complete prompt once and store each layer's unexpanded K/V tensors.
2. **Decode:** process only the newest token and attend to the stored prefix.

The cache is preallocated as `(batch, kv_heads, max_length, head_dim)`. RoPE receives the active
cache length as its absolute position offset. The two GQA K/V heads are expanded only temporarily
for SDPA; the cache itself never stores eight repeated heads.

No training or checkpoint changes were required.

## Correctness

- FP32 attention outputs match within `1e-5` for full, chunked, and token-by-token execution.
- The real 26M checkpoint has exact 400-token greedy sequence equality in FP32.
- BF16 cached and uncached greedy sequences match exactly through the 256-token benchmark.
- At 400 BF16 tokens, accumulated mixed-precision differences eventually select a different
  greedy branch. This is expected numerical drift between different kernel shapes, not evidence
  of a masking or position error.
- Existing learned-position, RoPE, MHA, GQA, training, and uncached inference paths remain valid.

An apparent earlier short-sequence mismatch was traced to comparing different RNG states at an
exact BF16 logit tie. `top_k=1` now uses deterministic `argmax` rather than multinomial sampling.

## Cache memory

For batch 1, context 512, BF16, eight layers, and head dimension 64:

| Attention | Stored K/V heads | Cache size |
|---|---:|---:|
| MHA | 8 | 8 MiB |
| GQA | 2 | **2 MiB** |

GQA reduces cache elements by 75%. These numbers describe cache tensors only, not model weights or
other activations.

## Benchmark protocol

- Checkpoint: `runs/tinystories-rope-swiglu-gqa/best.pt`, step 4,750
- Device: RTX 3060 12 GB
- Precision: BF16
- Prompt: `Once upon a time` (4 BPE tokens)
- Decoding: deterministic greedy
- Warmup: 16 tokens on each path
- Repetitions: 2; table reports medians
- Ollama model: unloaded during the measured run
- Structured result: `reports/results/tinystories-kv-cache-benchmark.json`

## Results

| New tokens | Uncached total | Cached total | Speedup | Uncached tok/s | Cached tok/s | Exact BF16 tokens |
|---:|---:|---:|---:|---:|---:|:---:|
| 64 | 0.450s | **0.437s** | 1.03× | 142.2 | **146.3** | yes |
| 128 | **0.943s** | 0.986s | 0.96× | **135.8** | 129.8 | yes |
| 256 | 2.044s | **2.020s** | 1.01× | 125.2 | **126.8** | yes |
| 400 | 3.284s | **2.999s** | **1.09×** | 121.8 | **133.4** | no; FP32 yes |

Peak allocated VRAM at 400 tokens decreased from 0.170 GiB to 0.157 GiB, about 7.2%.

## Interpretation

The cache is correct and reduces asymptotic attention work, but speedup is modest for this model:

1. At 26M parameters, one-token GPU operations are small and kernel-launch overhead is important.
2. Every step still computes the 8,000-class output projection.
3. Full-prefix operations can use larger, more efficient GPU kernels.
4. The current GQA path explicitly repeats K/V before SDPA.
5. Context is limited to 512, so the uncached quadratic cost has little time to dominate.

The cache becomes more valuable as model size, context length, batch size, or generated length
grows. The 400-token result already shows the expected direction: higher throughput and lower peak
memory.

## Limitations and next improvements

- Only two timing repetitions were used.
- Desktop and driver activity can affect short measurements.
- The cache does not yet support sliding-window eviction beyond 512 tokens.
- Batch rows that finish at different EOS positions are not independently retired.
- A native GQA-capable attention kernel could avoid temporary K/V expansion.
- CUDA graphs or compiled one-token decode could reduce launch overhead.

## Conclusion

KV caching is mathematically correct, lowers memory, and improves long-sequence decoding for the
current model, but it is not a dramatic speed win at this small scale. The main educational result
is understanding prefill versus decode and proving that GQA stores only two K/V heads. Cached
inference is now the interactive default; uncached inference remains available as a reference.
