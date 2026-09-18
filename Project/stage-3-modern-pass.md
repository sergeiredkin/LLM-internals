---
type: project
status: doing
---
# Stage 3 — Modern Architecture Pass

## Goal

Modernize the validated TinyStories GPT one feature at a time. For each feature: understand the
math, implement it independently, prove correctness with tests, run a controlled ablation, and
publish the result. See [[GPT-roadmap]].

## Tomorrow's scope (2026-09-17): RoPE only

Do **not** add SwiGLU, GQA, KV caching, or top-p sampling in the same experiment. The learned
position model in `reports/tinystories-baseline.md` is the control.

### 1. Learn the idea (30–45 minutes)

- [ ] Review why attention without position information is permutation-equivariant.
- [ ] Contrast learned absolute position embeddings with relative position information.
- [ ] Work through one 2D rotation pair by hand:
      `(x0, x1) -> (x0 cos θ - x1 sin θ, x0 sin θ + x1 cos θ)`.
- [ ] Explain why RoPE rotates queries and keys, but not values.
- [ ] Explain why `q_m · k_n` depends on relative displacement `n - m`.
- [x] Record the explanation and a numeric example in `Project/rope-learning-notes.md`.

**Learning gate:** be able to describe RoPE without referring to code.

### 2. Implement the smallest reusable component (45–60 minutes)

- [x] Add `RotaryEmbedding` and `rotate_half` in a focused module.
- [x] Precompute/cache cosine and sine values without trainable parameters.
- [x] Preserve device, dtype, and head shape `(batch, heads, sequence, head_dim)`.
- [x] Require an even attention head dimension.
- [x] Keep `position_encoding: learned` fully backward-compatible.

**Implementation gate:** existing learned-position checkpoints must still load and generate.

### 3. Prove the mathematics (30–45 minutes)

- [x] Test that position zero leaves vectors unchanged.
- [x] Test that rotation preserves each vector's L2 norm.
- [x] Test a hand-calculated 2D rotation.
- [x] Test relative-position dot-product invariance under a shared position shift.
- [x] Test shapes, BF16/FP32 behavior, and finite gradients.

**Correctness gate:** all old and new unit tests pass on CPU.

### 4. Integrate RoPE into GPT (30–45 minutes)

- [x] Apply RoPE to Q and K after splitting heads and before attention.
- [x] Do not rotate V.
- [x] Omit the learned position embedding table when RoPE is selected.
- [x] Keep causal SDPA behavior unchanged.
- [x] Verify parameter count decreases by exactly `512 × 512 = 262,144` for TinyStories.
- [x] Confirm future tokens still cannot affect earlier logits.

**Integration gate:** learned and RoPE models both pass forward, backward, causality, and generation
tests.

### 5. Run inexpensive behavior checks (30–45 minutes)

- [x] Run the one-batch overfit test with RoPE (`8.9944 -> 0.0000` in 300 steps).
- [x] Run a short GPU smoke train while Ollama remains available.
- [x] Check loss decreases, gradients remain finite, and VRAM stays within budget.
- [x] Save `configs/tinystories-rope-smoke.yaml` separately from the baseline config.

**Training gate:** do not launch the full run unless RoPE can overfit one batch and complete the
smoke run without instability.

### 6. Controlled full ablation (approximately 35–65 minutes)

- [x] Create `configs/tinystories-rope-shared.yaml` by changing only position encoding and output
      directory from the shared baseline.
- [x] Keep dataset, tokenizer, seed, optimizer, schedule, effective batch, context, and 5,000 steps
      identical.
- [x] Train with the shared-safe profile while the Ollama service remained available.
- [x] Select the best checkpoint using deterministic validation windows (step 4,750).
- [x] Evaluate with seed 2025 over the same 409,600 validation tokens.
- [x] Generate with the same prompt, seed 42, temperature 0.8, and top-k 40.

**Experiment gate:** no comparison claim until both models use the fixed evaluation protocol.

### 7. Report and publish (30 minutes)

- [x] Add the RoPE row to `reports/experiments.csv`.
- [x] Create `reports/tinystories-rope-ablation.md`.
- [x] Compare parameters, loss, perplexity, throughput, VRAM, generation speed, and sample quality.
- [x] Explain improvements, regressions, and experimental limitations honestly.
- [x] Export a compact BF16 inference checkpoint if the run is valid.
- [x] Commit and push source, tests, config, structured results, and report.
- [ ] Attach weights to a GitHub Release only after checksum verification.

## Baseline to beat or explain

| Metric | Learned-position baseline |
|---|---:|
| Parameters | 29,532,672 |
| Fixed validation loss | 1.9026 |
| Fixed perplexity | 6.70 |
| Evaluation throughput | 158,893 tokens/s |
| Generation throughput (no cache) | 182.3 tokens/s |
| Peak inference VRAM | 0.60 GiB |
| Full training runtime | 33m 7s |

RoPE does not need to win every metric to be useful. A valid result may show similar loss with
fewer parameters, different optimization behavior, or better length generalization. Record what
happens rather than selecting only favorable evidence.

## Later work — not tomorrow's first experiment

- [x] SwiGLU controlled ablation
- [x] Grouped-query attention controlled ablation
- [x] KV-cache generation and latency benchmark
- [x] Top-p sampling comparison
- [ ] LoRA/QLoRA
- [x] Per-channel INT8 quantization
- [x] Packed groupwise INT4 quantization

## Log

- 2026-09-16: deterministic learned-position baseline and public inference release completed.
- 2026-09-16: added the RoPE derivation, numeric examples, and learning questions.
- 2026-09-16: implemented and tested isolated RoPE primitives. All 30 tests pass; the GPT and its
  checkpoints remain unchanged.
- 2026-09-16: integrated RoPE into Q/K while preserving learned positions as the default. The
  full RoPE model has 29,270,528 parameters, passes BF16 forward/backward, and all 32 tests pass.
  The published learned-position checkpoint still loads strictly.
- 2026-09-16: RoPE one-batch overfit passed on CPU: loss 8.9944 to 0.0000 in 300 steps.
- 2026-09-16: full-size 20-step BF16 smoke test passed while Ollama held 6.53 GiB VRAM.
  Training loss fell 9.0263 to 6.1024, validation loss fell 9.0389 to 6.6950, typical throughput
  was 25k–29k tokens/s, and peak PyTorch memory was 1.11 GiB reserved.
- 2026-09-17: completed the controlled 5,000-step RoPE run. Fixed validation loss improved from
  1.9026 to 1.8401 and perplexity from 6.70 to 6.30 while removing 262,144 parameters. Best
  checkpoint was step 4,750; peak training memory was 1.11 GiB reserved.
- 2026-09-17: completed [[stage-3-swiglu]]. Fixed loss improved slightly to 1.8316, with slower
  inference and a worse fixed sample; the report treats the result as mixed rather than decisive.
- 2026-09-17: started [[stage-3-gqa]] by isolating and validating 8-query/2-KV-head sharing.
- 2026-09-18: completed [[stage-3-gqa]]. It removed 10.77% of cumulative model parameters for a
  0.33% fixed-loss increase and a 7.05% back-to-back full-sequence speed improvement.
- 2026-09-18: completed [[stage-3-kv-cache]]. The 2 MiB GQA cache is 75% smaller than MHA;
  cached decoding was 1.09x faster with 7.2% lower peak allocation at 400 generated tokens.
- 2026-09-18: completed [[stage-3-top-p]] with isolated filtering tests and a fixed-seed top-k,
  top-p, and combined comparison. The interactive default is now top-k 40 plus top-p 0.9.
- 2026-09-18: completed [[stage-5-int8-quantization]]. Quantizing 56 hidden linear layers changed
  fixed validation loss by only +0.000071 and reduced the BF16 export from 50.3 to 29.5 MiB.
- 2026-09-18: completed [[stage-5-int4-quantization]]. Group-64 packed INT4 reduced the export to
  20.3 MiB with a +0.010405 fixed-loss change; group sizes 32/64/128 were compared.
