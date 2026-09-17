# TinyStories SwiGLU Controlled Ablation

**Date:** 2026-09-17

**Question:** Does a parameter-matched SwiGLU improve the RoPE TinyStories model when every other
training setting remains fixed?

## Experimental control

The experiment changes only the feed-forward network:

```text
GELU:   down(GELU(up(x)))
SwiGLU: down(SiLU(gate(x)) * value(x))
```

The GELU hidden width is 2,048. SwiGLU uses 1,360—the nearest multiple of 16 to two thirds of
2,048—because SwiGLU has three matrices instead of two. Dataset, tokenizer, split, seed, RoPE,
model dimensions, context, optimizer, schedule, effective batch, precision, and 5,000-step budget
remain fixed.

## Reproduction identity

- Config: `configs/tinystories-rope-swiglu-shared.yaml`
- Best checkpoint: `runs/tinystories-rope-swiglu/best.pt`, step 4,750
- Full checkpoint SHA-256: `6d1c057bf785a8451da5dcd672a0b92bfa303c52b664c55a3615326fe6132096`
- Compact BF16 export: `exports/tinystories-29m-rope-swiglu-bf16.pt`
- Export size: 56.3 MiB
- Export SHA-256: `3daf2829874a67fa158983c673037982fb96bc77fe0e1d67dc803126b21bb422`
- Training seed: 1337
- Fixed evaluation seed: 2025
- Fixed generation seed: 42
- Machine-readable result: `reports/results/tinystories-rope-swiglu.json`

## Training result

| Metric | RoPE + SwiGLU |
|---|---:|
| Parameters | 29,204,992 |
| Steps | 5,000 |
| Best online deterministic validation loss | 1.8795 |
| Best checkpoint | 4,750 |
| Typical training throughput | 38k–41k tokens/s |
| Peak PyTorch VRAM | 1.19 GiB reserved |
| Exact wall time | 34m 59s (2,099.7 seconds) |

The run launched while Ollama held approximately 6.53 GiB VRAM. Ollama later unloaded its model;
the service remained available throughout.

## Fixed evaluation comparison

Both checkpoints used seed 2025 and the same 409,600 validation tokens.

| Metric | RoPE + GELU control | RoPE + SwiGLU | Change |
|---|---:|---:|---:|
| Parameters | 29,270,528 | **29,204,992** | -65,536 (-0.22%) |
| Fixed validation loss | 1.8401 | **1.8316** | -0.0085 (-0.46%) |
| Fixed perplexity | 6.297 | **6.244** | -0.053 (-0.85%) |
| Evaluation throughput | **156,663** | 138,625 tok/s | -11.5% |
| Peak inference VRAM | **0.60 GiB** | 0.62 GiB | +0.02 GiB |
| Generation throughput, no cache | **140.1** | 125.2 tok/s | -10.6% |

SwiGLU achieved a small loss improvement with slightly fewer parameters. Its extra gate projection
and elementwise product reduced throughput and increased activation memory despite the narrower
hidden layer.

## Fixed-seed sample

Prompt: `Once upon a time, there was a little fox named Pip.`

> Once upon a time, there was a little fox named Pip. Pip loved to play hide and seek with his
> animal friends. One day, Poppy and his friends wanted to have a race.
>
> Zoom and his friends tried to do something hard. But they were too clever together. They did not
> know what to do. They decided to ask Zackers for help.
>
> Zip had an idea. He said, "Let's play hide and seek. I will hide behind the trees and then you
> will find me." But Zack did not want to give up. He thought it was a good idea.

The story has recognizable structure and dialogue, but changes names repeatedly and is less
coherent than the fixed RoPE + GELU sample. This demonstrates why a small loss improvement does not
guarantee that every sampled story will look better.

## Findings

1. Parameter-matched SwiGLU improved fixed loss by 0.0085 and perplexity by 0.053.
2. The improvement is modest and comes from one seed, so it may be within run-to-run variation.
3. SwiGLU removed 65,536 parameters from the full model but increased activation memory by about
   0.02 GiB.
4. Evaluation and uncached generation were approximately 11% slower.
5. The fixed generated sample was qualitatively worse than the RoPE + GELU sample.
6. SwiGLU works correctly and is useful educationally, but this experiment does not establish a
   decisive quality advantage at 29M parameters.

## Limitations

- Only one training seed was run.
- Human judgment is based on one fixed sample, not a blinded prompt suite.
- Generation uses no KV cache.
- Kernel efficiency may depend on the chosen hidden width and GPU.
- The loss difference is small enough that repeated runs would be needed for statistical
  confidence.

## Conclusion

SwiGLU produced the best numerical validation result so far—loss 1.8316 and perplexity 6.24—with
slightly fewer parameters, but it was slower and its fixed sample was less coherent. We retain the
implementation and can use it in the cumulative modern model, while treating its quality advantage
as provisional rather than proven.
