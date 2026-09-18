---
type: project
status: doing
---
# Stage 3 — KV-Cache Generation

## Goal

Make autoregressive generation reuse keys and values from earlier tokens instead of recomputing the
entire prefix at every step. Prove cached and uncached decoding agree before benchmarking speed.
No model retraining is required.

## Mental model

Without a cache, generating one token repeatedly processes all prior tokens:

```text
step 1: process T prompt tokens
step 2: process T + 1 tokens
step 3: process T + 2 tokens
```

With a cache:

```text
prefill: process T prompt tokens and store their K/V tensors
step 1: process one new token and attend to stored K/V
step 2: process one new token and attend to stored K/V
```

Queries are needed only for the current token. Previous keys and values do not change and can be
reused.

## GQA memory advantage

The cumulative model has eight query heads and two K/V heads. Cache tensors retain the two
unexpanded K/V heads:

```text
shape per layer = (batch, 2 KV heads, sequence, 64 head dimensions)
```

At batch 1, context 512, BF16, and eight layers:

```text
GQA cache: 8 × 2(K/V) × 1 × 2 × 512 × 64 × 2 bytes = 4 MiB
MHA cache: 8 × 2(K/V) × 1 × 8 × 512 × 64 × 2 bytes = 16 MiB
```

GQA therefore reduces cache elements by 75%.

## Checklist

### Isolated cache

- [x] Add fixed-capacity per-layer K/V storage.
- [x] Store tensors as `(batch, kv_heads, sequence, head_dim)`.
- [x] Append prompt chunks and individual decode positions.
- [x] Track active length, remaining capacity, and memory bytes.
- [x] Reject shape, dtype, device, and capacity mismatches.
- [x] Reset without reallocating storage.
- [x] Verify an 8-KV-head cache is four times larger than a 2-KV-head cache.

### Attention integration

- [x] Accept an optional layer cache and position offset in attention.
- [x] Apply RoPE using the absolute cache position.
- [x] Append unexpanded K/V before temporary GQA head expansion.
- [x] Use a causal mask correctly for multi-token prefill and incremental decode.
- [x] Verify cached and uncached attention outputs agree.

### Model integration

- [ ] Maintain one cache per transformer layer.
- [ ] Implement prompt prefill followed by one-token decoding.
- [ ] Verify cached and uncached logits agree at every position.
- [ ] Verify greedy generated token sequences match exactly.
- [ ] Preserve the existing uncached path and checkpoint compatibility.

### Benchmark and report

- [ ] Benchmark multiple prompt and generation lengths.
- [ ] Report prefill time separately from decode time.
- [ ] Measure tokens/s, peak VRAM, and cache memory.
- [ ] Compare GQA and MHA cache sizes.
- [ ] Update reports and publish code/results.

## Log

- 2026-09-18: implemented the isolated preallocated per-layer cache. GPT and attention behavior
  remained unchanged until cache storage tests passed.
- 2026-09-18: integrated optional per-layer caching into attention. Token-by-token and chunked
  cached outputs match full causal attention; maximum observed FP32 difference was 1.79e-7. The
  cache retains two unexpanded GQA heads. All 57 tests pass.
