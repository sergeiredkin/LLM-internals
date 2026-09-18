---
type: project
status: done
---
# Stage 3 — Top-p (Nucleus) Sampling

## Goal

Add adaptive probability-mass filtering without changing model weights, checkpoint compatibility,
or the existing default generation API.

## Core idea

Top-k always keeps a fixed number of candidates. Top-p instead sorts token probabilities from
largest to smallest and keeps the smallest prefix whose cumulative probability reaches `p`.

For sorted probabilities `q_1 >= q_2 >= ...`, retain token `i` while:

`sum(q_j for j < i) < p`

This condition keeps the token that reaches or crosses `p`. It also handles an exact boundary
correctly: if retained mass already equals `p`, the next token is excluded.

## Order of operations

1. Divide logits by temperature.
2. Optionally retain exactly the top-k logits.
3. Compute top-p probabilities from the remaining logits.
4. Retain the smallest nucleus reaching `p`.
5. Renormalize implicitly with softmax and sample.

With `top_k=40, top_p=0.9`, the candidate set can never exceed 40 tokens but may shrink when the
model is confident. `top_p=1.0` disables nucleus filtering. `top_k=1` remains deterministic argmax.

## Correctness checklist

- [x] Keep the smallest set reaching the requested mass.
- [x] Always retain at least the most likely token.
- [x] Restore original vocabulary order after sorting.
- [x] Handle independent batch rows.
- [x] Handle exact cumulative-probability boundaries.
- [x] Compose top-k and top-p filters.
- [x] Reproduce stochastic samples with a fixed generator seed.
- [x] Reject invalid shapes, temperatures, k values, and p values.
- [x] Preserve cached and uncached generation.
- [x] Preserve old checkpoints and default API behavior.

## Result

The implementation lives in `llm/sampling.py`, separate from the GPT architecture. Generation and
the CLI accept an optional `top_p`; no retraining is needed. All 74 repository tests pass.

The fixed comparison uses the 26M RoPE + SwiGLU + GQA checkpoint, prompt `Once upon a time`, seed
42, temperature 0.8, and cached generation. See `reports/tinystories-top-p-comparison.md`.
