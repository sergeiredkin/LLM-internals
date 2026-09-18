---
type: project
status: done
---
# Stage 3 — Grouped-Query Attention Controlled Ablation

## Question

Can query heads share fewer key/value heads without materially hurting TinyStories validation
quality, while reducing parameters and future KV-cache size?

## Mental model

Multi-head attention (MHA) gives every query head its own K and V head:

```text
8 query heads -> 8 key heads + 8 value heads
```

The planned grouped-query attention (GQA) model keeps all query heads but uses two K/V heads:

```text
8 query heads -> 2 key heads + 2 value heads

Q0 Q1 Q2 Q3 share KV0
Q4 Q5 Q6 Q7 share KV1
```

Queries remain specialized. Only the information used to match and retrieve content is shared.
GQA lies between MHA and multi-query attention:

```text
MHA: 8 Q / 8 KV
GQA: 8 Q / 2 KV
MQA: 8 Q / 1 KV
```

## Expected parameter effect

For `d_model=512`, eight query heads, and head dimension 64:

- MHA K/V width: `8 × 64 = 512`
- GQA K/V width: `2 × 64 = 128`
- Reduction per transformer block: `2 × 512 × (512 - 128) = 393,216`
- Reduction over eight blocks: `3,145,728`
- Expected cumulative model size: `29,204,992 - 3,145,728 = 26,059,264`

A future KV cache can store two K/V heads instead of eight, reducing its K/V elements by 75%.
The current training implementation explicitly repeats K/V heads before SDPA, so it proves the
GQA mathematics and reduces projection parameters, but does not yet realize the full attention
activation-memory benefit. The cache implementation must retain unexpanded K/V tensors.

## Control

The control is RoPE + SwiGLU:

- Parameters: 29,204,992
- Fixed validation loss: 1.8316
- Fixed perplexity: 6.24
- Evaluation throughput: 138,625 tokens/s
- Generation throughput without cache: 125.2 tokens/s

The GQA experiment will change only `n_kv_heads: 8 -> 2`, plus experiment name and output path.

## Checklist

### Isolated behavior and tests

- [x] Extract and document the K/V head-sharing operation.
- [x] Verify each K/V head is assigned to the correct query-head group.
- [x] Verify K/V projection widths shrink from 512 to 128.
- [x] Verify the exact attention parameter-reduction formula.
- [x] Verify forward shape, backward pass, and finite gradients.
- [x] Preserve MHA behavior when `n_kv_heads == n_heads`.

### Full model checks

- [x] Verify the cumulative model has exactly 26,059,264 parameters.
- [x] Verify RoPE is applied before K/V head expansion.
- [x] Verify causality and generation with 8 Q / 2 KV heads.
- [x] Pass one-batch overfitting with preserved 4 Q / 1 KV sharing (`8.9890 -> 0.0000`).
- [x] Pass a 20-step full-size BF16 GPU smoke test.

### Controlled experiment

- [x] Create a 5,000-step controlled GQA configuration.
- [x] Train and select the best deterministic-validation checkpoint (step 4,750).
- [x] Evaluate the same fixed 409,600 tokens with seed 2025.
- [x] Generate with the same prompt and seed 42.
- [x] Report quality, speed, memory, parameters, and limitations.
- [x] Commit and push code, tests, configuration, results, and report.

## Log

- 2026-09-17: existing GQA path was isolated and explicitly tested. The implementation uses
  grouped K/V projections and deterministic head expansion before PyTorch SDPA.
- 2026-09-17: full cumulative model has the predicted 26,059,264 parameters and passes BF16
  forward/backward with finite gradients.
- 2026-09-17: the first overfit invocation exposed that the educational harness forced MHA. The
  harness was corrected and regression-tested to preserve the Q/KV sharing ratio. The valid GQA
  run used 4 Q / 1 KV heads and memorized the batch from loss 8.9890 to 0.0000.
- 2026-09-17: full-size 8 Q / 2 KV BF16 smoke test passed while Ollama held 6.53 GiB VRAM.
  Training loss fell 9.0394 to 6.4544, validation loss fell 9.0512 to 6.9138, typical throughput
  reached 28k–30k tokens/s, gradients remained finite, and peak PyTorch memory was 1.14 GiB
  reserved.
- 2026-09-18: controlled 5,000-step run completed in 34m 25s. GQA removed 3,145,728 parameters
  and improved back-to-back evaluation speed by 7.05%, while fixed loss increased slightly from
  1.8316 to 1.8376. The result is a favorable efficiency trade.
