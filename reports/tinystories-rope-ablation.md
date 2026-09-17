# TinyStories RoPE Controlled Ablation

**Date:** 2026-09-17

**Question:** What changes when learned absolute position embeddings are replaced by RoPE, while
every other training setting remains fixed?

## Experimental control

The RoPE configuration differs from `configs/tinystories-shared.yaml` only in:

- experiment name;
- `position_encoding: rope` instead of `learned`;
- checkpoint output directory.

Dataset, tokenizer, document split, seed, architecture dimensions, context, optimizer, schedule,
effective batch, precision, and 5,000-step budget are unchanged.

## Reproduction identity

- Config: `configs/tinystories-rope-shared.yaml`
- Best checkpoint: `runs/tinystories-rope/best.pt`, step 4,750
- Full checkpoint SHA-256: `a711b6dc1c109f570030dd9211d72c5f7b610834f5f1a819501a6d4072678da2`
- Compact BF16 export: `exports/tinystories-29m-rope-bf16.pt`
- Export size: 56.4 MiB
- Export SHA-256: `06936b123ec9e8e107652e71cba40472ae45b6271954d4d51e55e7ecc6827d7c`
- Training seed: 1337
- Fixed evaluation seed: 2025
- Fixed generation seed: 42
- Machine-readable result: `reports/results/tinystories-rope.json`

## Training result

| Metric | RoPE |
|---|---:|
| Parameters | 29,270,528 |
| Steps | 5,000 |
| Best online deterministic validation loss | 1.8946 |
| Best checkpoint | 4,750 |
| Typical training throughput | 40k–42k tokens/s |
| Peak PyTorch VRAM | 1.11 GiB reserved |
| Runtime | approximately 34 minutes |

The runtime is estimated from throughput because elapsed wall-clock time was not yet emitted by
the trainer. It should not be treated as an exact benchmark.

## Fixed evaluation comparison

Both checkpoints were evaluated on the same 100 batches, batch size 8, context 512, and seed
2025: exactly 409,600 validation tokens.

| Metric | Learned positions | RoPE | Change |
|---|---:|---:|---:|
| Parameters | 29,532,672 | **29,270,528** | -262,144 (-0.89%) |
| Fixed validation loss | 1.9026 | **1.8401** | -0.0626 (-3.29%) |
| Fixed perplexity | 6.70 | **6.30** | -0.41 (-6.06%) |
| Evaluation throughput | **158,893** | 156,663 tok/s | -1.40% |
| Peak inference VRAM | 0.60 GiB | 0.60 GiB | no measured change |
| Generation throughput | **182.3** | 140.1 tok/s | -23.2% |

RoPE achieved lower loss and perplexity with fewer trainable parameters. Evaluation throughput
changed only slightly. Uncached token-by-token generation was slower because every layer now
applies rotations while repeatedly recomputing the prefix. This is a single short generation
measurement, so a dedicated repeated latency benchmark is needed before making a strong speed
claim.

## Fixed-seed RoPE sample

Prompt: `Once upon a time, there was a little fox named Pip.`

> Once upon a time, there was a little fox named Pip. Pip loved to play hide and seek with his
> friends. One day, Pip saw a group of animals playing together. They were wearing colorful
> dresses and shiny shoes. Pip wanted to join them, but he was too afraid to ask.
>
> He asked his friends to help him. They all started to walk, but Pip was not happy. He wanted to
> be friends. So, he decided to go to the group and play. In the pile of clothes, he found a group
> of kids playing together. The group of kids were laughing and having fun.
>
> A little bird saw Pip and the kids playing together. The little bird was happy too. It flew high
> in the sky, singing with all its new friends. Pip and the group played some more. They had so
> much fun that they wanted to stay with them forever.

The RoPE sample maintains Pip's name throughout, unlike the baseline fixed sample, but one sample
is anecdotal and cannot establish a general quality improvement.

## Findings

1. The controlled RoPE model improved fixed validation loss by 0.0626 and perplexity by 0.41.
2. Removing the learned `512 × 512` position table saved exactly 262,144 parameters.
3. Training remained stable in BF16 and used essentially the same memory as the baseline.
4. Full-sequence evaluation cost changed little; uncached autoregressive generation was slower.
5. The fixed sample appears more consistent, but broader prompt-based evaluation is required.
6. The result supports keeping RoPE for the next controlled architecture experiment.

## Limitations

- One training seed cannot measure run-to-run variance.
- Online validation procedures differed: the original baseline used random windows during
  training, while RoPE used deterministic windows. The final comparison avoids this issue by
  evaluating both saved checkpoints with the same fixed protocol.
- Exact RoPE training wall time was not instrumented.
- Generation speed is based on one prompt and includes no KV cache.
- No context-length extrapolation experiment was performed.

## Conclusion

For this 29M TinyStories model and fixed 5,000-step budget, RoPE is the better position encoding:
it produced lower held-out loss with fewer parameters at almost unchanged full-sequence inference
cost. The generation latency regression should be revisited after KV caching and repeated timing.
