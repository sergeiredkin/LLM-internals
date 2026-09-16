# TinyStories Baseline Report — Learned Positions

**Date:** 2026-09-16

**Purpose:** establish the fixed reference used for all Stage 3 architecture ablations.

## Reproduction identity

- Source checkpoint: `runs/tinystories/best.pt`
- Checkpoint step: 4,250
- Full checkpoint SHA-256: `c1a8c5579f795ed1b5aea1f2d21a71ec761270163af441800b8f16886edb96ae`
- Inference export: `exports/tinystories-29m-bf16.pt`
- Inference export size: 56.9 MiB
- Inference export SHA-256: `3fc7fe23071e2ae76256c8b3370c832b23d5aef6bd9583afc69addef0c46b111`
- Training seed: 1337
- Fixed evaluation seed: 2025
- Fixed generation seed: 42
- Machine-readable result: `reports/results/tinystories-learned-baseline.json`

The full checkpoint contains optimizer and resume state and remains local. The compact BF16
export contains model weights, configuration, and tokenizer, but no optimizer state.

## Model

| Setting | Value |
|---|---:|
| Architecture | Decoder-only, pre-norm GPT |
| Position encoding | Learned absolute embedding |
| Parameters | 29,532,672 |
| Layers | 8 |
| Width | 512 |
| Attention heads | 8 |
| KV heads | 8 |
| Context length | 512 |
| Vocabulary | 8,000 byte-level BPE tokens |
| Training precision | BF16 |

## Data and training

| Metric | Value |
|---|---:|
| Training documents | 50,000 |
| Validation documents | 5,000 |
| Training tokens | 10,970,105 |
| Validation tokens | 988,909 |
| Training steps | 5,000 |
| Tokens per update | 16,384 |
| Training runtime | 33m 7s |
| Training peak PyTorch VRAM | 1.12 GiB reserved |
| Best checkpoint step | 4,250 |
| Online sampled validation loss | 1.8759 |

The online value was measured during the original run with random validation windows. Future
experiments use the fixed protocol below instead.

## Fixed evaluation protocol

Every architecture is evaluated on exactly the same windows:

- Seed: 2025
- Batches: 100
- Batch size: 8
- Context length: 512
- Evaluated tokens: 409,600

| Metric | Baseline |
|---|---:|
| Fixed validation loss | **1.9026** |
| Fixed validation perplexity | **6.70** |
| Evaluation throughput | 158,893 tokens/s |
| Peak inference VRAM | 0.60 GiB |

## Fixed generation protocol

- Prompt: `Once upon a time, there was a little fox named Pip.`
- Seed: 42
- Temperature: 0.8
- Top-k: 40
- Maximum new tokens: 200
- Generated tokens before EOS: 169
- Generation throughput without KV cache: 182.3 tokens/s

### Sample

> Once upon a time, there was a little fox named Pip. Pip loved to play hide and seek
> with his friends. One day, Pip saw a big tree with lots of leaves. He wanted to find a
> friend to play with, but he didn't have any friends to play together.
>
> Pinky decided to jump over the tree and ask his friends for help. The friends were happy
> and said, "No, you can't play here. You're too small." But Kitty did not listen and
> grabbed a big leaf.
>
> Pip's friends came over and said, "You are being mean to each other. If you don't share
> the leaves, you will get everyone you want." So, Kitty agreed to share the leaves with
> his friends. They all played together and had a great time.

The story has coherent form, dialogue, and a conclusion, but inconsistent character names and
logic. These are useful qualitative limitations to track in later experiments.

## Findings

1. BPE reduced the example text from 71 characters to 17 tokens, allowing much longer semantic
   spans than the character model at the same context length.
2. The model learned story structure and dialogue from only 50,000 documents.
3. Fixed-window loss (1.9026) differs from the online sampled estimate (1.8759), demonstrating
   why architecture comparisons require identical evaluation windows.
4. Generation is already coherent, but factual continuity and character identity remain weak.
5. Generation recomputes the whole prefix and reaches 182 tokens/s. KV caching should provide a
   clear speed comparison later.
6. The learned position table contains 262,144 parameters. RoPE should remove this table while
   changing how Q/K represent position.

## Controlled ablation policy

For RoPE and later features, keep these fixed unless the report explicitly says otherwise:

- Dataset and tokenizer
- Train/validation documents
- Seed and evaluation windows
- Layers, width, heads, context, and effective batch
- Optimizer, schedule, steps, and sampling settings

Report parameter count, fixed validation loss, perplexity, training/inference throughput, peak
VRAM, generation speed, and the fixed-seed sample. Change one architectural feature per run.
