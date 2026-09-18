# TinyStories Top-p Sampling Comparison

**Date:** 2026-09-18

## Protocol

Sampling changes inference only; all modes use identical model weights.

- Checkpoint: `runs/tinystories-rope-swiglu-gqa/best.pt`, step 4,750
- Prompt: `Once upon a time`
- Seed: 42
- Temperature: 0.8
- Maximum new tokens: 200
- Cached CUDA BF16 generation

| Mode | top-k | top-p |
|---|---:|---:|
| Fixed top-k | 40 | disabled |
| Adaptive nucleus | disabled | 0.9 |
| Combined | 40 | 0.9 |

## What changes

- **Top-k 40** always permits up to 40 candidates, even when probability outside the strongest few
  tokens is negligible.
- **Top-p 0.9** adapts: a confident prediction may retain only a few tokens, while an uncertain
  prediction may retain many.
- **Combined** first limits the candidate pool to 40 and then retains enough of that pool to reach
  90% of its renormalized mass.

## Fixed-seed observations

### Top-k 40

The story remained structurally complete, but introduced the odd event that a clay castle was
thrown into the air and shrank. It also repeated “clay castle” heavily.

### Top-p 0.9

The story maintained a build-and-share theme, although it switched between a pile, house, and
tower and included the strange phrase “blowing the blocks.”

### Top-k 40 plus top-p 0.9

The story had the clearest event progression—exploration, discovery, opening a box, and finding a
toy car—but still produced repetition and “make his new car a nice gift for his new car.”

These are single-seed qualitative observations, not evidence that one method is universally
better. TinyStories samples remain limited by the 26M model and short training run.

## Reproduction

Top-k only:

```bash
python -m scripts.generate --checkpoint runs/tinystories-rope-swiglu-gqa/best.pt \
  --prompt 'Once upon a time' --max-new-tokens 200 --temperature 0.8 \
  --top-k 40 --seed 42 --device cuda --use-kv-cache
```

Top-p only:

```bash
python -m scripts.generate --checkpoint runs/tinystories-rope-swiglu-gqa/best.pt \
  --prompt 'Once upon a time' --max-new-tokens 200 --temperature 0.8 \
  --top-k 0 --top-p 0.9 --seed 42 --device cuda --use-kv-cache
```

Combined:

```bash
python -m scripts.generate --checkpoint runs/tinystories-rope-swiglu-gqa/best.pt \
  --prompt 'Once upon a time' --max-new-tokens 200 --temperature 0.8 \
  --top-k 40 --top-p 0.9 --seed 42 --device cuda --use-kv-cache
```

## Decision

Retain all three modes. The interactive launcher uses `top_k=40, top_p=0.9` as a conservative
adaptive default. Passing `--top-k 0` gives pure nucleus sampling, while omitting `--top-p`
preserves the earlier top-k behavior.
