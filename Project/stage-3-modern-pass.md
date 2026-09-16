---
type: project
status: todo
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
- [ ] Record the explanation and a numeric example in `Project/rope-learning-notes.md`.

**Learning gate:** be able to describe RoPE without referring to code.

### 2. Implement the smallest reusable component (45–60 minutes)

- [ ] Add `RotaryEmbedding` and `rotate_half` in a focused module.
- [ ] Precompute/cache cosine and sine values without trainable parameters.
- [ ] Preserve device, dtype, and head shape `(batch, heads, sequence, head_dim)`.
- [ ] Require an even attention head dimension.
- [ ] Keep `position_encoding: learned` fully backward-compatible.

**Implementation gate:** existing learned-position checkpoints must still load and generate.

### 3. Prove the mathematics (30–45 minutes)

- [ ] Test that position zero leaves vectors unchanged.
- [ ] Test that rotation preserves each vector's L2 norm.
- [ ] Test a hand-calculated 2D rotation.
- [ ] Test relative-position dot-product invariance under a shared position shift.
- [ ] Test shapes, BF16/FP32 behavior, and finite gradients.

**Correctness gate:** all old and new unit tests pass on CPU.

### 4. Integrate RoPE into GPT (30–45 minutes)

- [ ] Apply RoPE to Q and K after splitting heads and before attention.
- [ ] Do not rotate V.
- [ ] Omit the learned position embedding table when RoPE is selected.
- [ ] Keep causal SDPA behavior unchanged.
- [ ] Verify parameter count decreases by exactly `512 × 512 = 262,144` for TinyStories.
- [ ] Confirm future tokens still cannot affect earlier logits.

**Integration gate:** learned and RoPE models both pass forward, backward, causality, and generation
tests.

### 5. Run inexpensive behavior checks (30–45 minutes)

- [ ] Run the one-batch overfit test with RoPE.
- [ ] Run a short GPU smoke train while Ollama remains available.
- [ ] Check loss decreases, gradients remain finite, and VRAM stays within budget.
- [ ] Save `configs/tinystories-rope-smoke.yaml` separately from the baseline config.

**Training gate:** do not launch the full run unless RoPE can overfit one batch and complete the
smoke run without instability.

### 6. Controlled full ablation (approximately 35–65 minutes)

- [ ] Create `configs/tinystories-rope-shared.yaml` by changing only position encoding and output
      directory from the shared baseline.
- [ ] Keep dataset, tokenizer, seed, optimizer, schedule, effective batch, context, and 5,000 steps
      identical.
- [ ] Train while sharing the GPU with Ollama, unless a faster exclusive-GPU window is approved.
- [ ] Select the best checkpoint using deterministic validation windows.
- [ ] Evaluate with seed 2025 over the same 409,600 validation tokens.
- [ ] Generate with the same prompt, seed 42, temperature 0.8, and top-k 40.

**Experiment gate:** no comparison claim until both models use the fixed evaluation protocol.

### 7. Report and publish (30 minutes)

- [ ] Add the RoPE row to `reports/experiments.csv`.
- [ ] Create `reports/tinystories-rope-ablation.md`.
- [ ] Compare parameters, loss, perplexity, throughput, VRAM, generation speed, and sample quality.
- [ ] Explain improvements, regressions, and experimental limitations honestly.
- [ ] Export a compact BF16 inference checkpoint if the run is valid.
- [ ] Commit and push source, tests, config, structured results, and report.
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

- [ ] SwiGLU controlled ablation
- [ ] Grouped-query attention controlled ablation
- [ ] KV-cache generation and latency benchmark
- [ ] Top-p sampling comparison
- [ ] LoRA/QLoRA
- [ ] INT4 quantization

## Log

- 2026-09-16: deterministic learned-position baseline and public inference release completed.
