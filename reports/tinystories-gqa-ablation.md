# TinyStories GQA Controlled Ablation

**Date:** 2026-09-17

**Question:** Can four query heads share each K/V head while retaining most model quality and
reducing parameters, memory, checkpoint size, and future KV-cache size?

## Experimental control

The experiment changes only:

```text
n_kv_heads: 8 -> 2
```

All eight query heads remain. Q0–Q3 share KV0 and Q4–Q7 share KV1. Dataset, tokenizer, split,
seed, RoPE, SwiGLU, model width/depth, optimizer, schedule, effective batch, precision, context,
and 5,000-step budget remain fixed.

## Reproduction identity

- Config: `configs/tinystories-rope-swiglu-gqa-shared.yaml`
- Best checkpoint: `runs/tinystories-rope-swiglu-gqa/best.pt`, step 4,750
- Full checkpoint SHA-256: `949eb54da8689369689310d19e0d877bc762166b9f5f8b721263f483af7ad189`
- Compact BF16 export: `exports/tinystories-26m-rope-swiglu-gqa-bf16.pt`
- Export size: 50.3 MiB
- Export SHA-256: `83ca63bf3884b238df4b36ba630500a72bc7030dea12ef89b4cd2881910e5b01`
- Training seed: 1337
- Fixed evaluation seed: 2025
- Fixed generation seed: 42

## Training result

| Metric | RoPE + SwiGLU + GQA |
|---|---:|
| Parameters | 26,059,264 |
| Steps | 5,000 |
| Best online deterministic validation loss | 1.8888 |
| Best checkpoint | 4,750 |
| Typical uncongested training throughput | 41k–43k tokens/s |
| Peak PyTorch VRAM | 1.14 GiB reserved |
| Exact wall time | 34m 25s (2,065.4 seconds) |
| Full checkpoint size | 312.8 MB |

Ollama remained available and reloaded during parts of training. Throughput temporarily fell to
approximately 26k–28k tokens/s during shared-GPU contention, then recovered.

## Fixed quality comparison

Both models were evaluated on the same 409,600 validation tokens with seed 2025.

| Metric | 8 Q / 8 KV control | 8 Q / 2 KV GQA | Change |
|---|---:|---:|---:|
| Parameters | 29,204,992 | **26,059,264** | -3,145,728 (-10.77%) |
| Fixed validation loss | **1.8316** | 1.8376 | +0.0060 (+0.33%) |
| Fixed perplexity | **6.244** | 6.282 | +0.038 (+0.61%) |
| Training peak VRAM | 1.19 GiB | **1.14 GiB** | -0.05 GiB |
| Full checkpoint size | 350.6 MB | **312.8 MB** | -37.8 MB |
| Training wall time | 34m 59s | **34m 25s** | -34 seconds |

GQA gives up a small amount of validation quality while removing more than 10% of all model
parameters. The quality regression is much smaller than the parameter reduction.

## Back-to-back shared-GPU speed comparison

The first GQA evaluation ran while Ollama was loaded, whereas the historical control timing was
recorded with Ollama unloaded. To avoid that confound, both checkpoints were benchmarked
back-to-back under the same current shared-GPU condition. Structured results are in
`reports/results/tinystories-gqa-shared-speed.json`.

| Metric | 8 Q / 8 KV control | 8 Q / 2 KV GQA | Change |
|---|---:|---:|---:|
| Full-sequence evaluation | 140,390 tok/s | **150,293 tok/s** | +7.05% |
| Uncached generation | **123.9 tok/s** | 120.6 tok/s | -2.70% |
| Peak inference VRAM | 0.623 GiB | **0.600 GiB** | -3.76% |

The current implementation explicitly repeats K/V heads before standard SDPA. This preserves GQA
mathematics and shrinks projections, but does not realize all possible attention-memory savings.
The future KV cache should store the two unexpanded K/V heads, reducing cache elements by 75%.

## Fixed-seed sample

> Once upon a time, there was a little fox named Pip. Pip was very shy and didn't like to play
> with other animals. One day, Fin saw a big, scary bear coming towards him. The bear was very
> afraid and wanted to escape. But Bobo was brave and said no.
>
> The bear got scared and started to run. The bear was very scared and didn't know what to do. But
> then, he remembered that he was brave and knew he had to find some help.
>
> Bree was very safe and happy. He thanked the bear for saving him from the scary bear. From that
> day on, Bobo never went into the forest again.

The output has story structure and a conclusion, but still changes character names and contains
logical inconsistencies. It is not clearly better or worse than the MHA control from one sample.

## Findings

1. GQA removed 3.15M parameters—10.77% of the cumulative model—for only a 0.33% loss increase.
2. Perplexity increased modestly from 6.244 to 6.282.
3. Full-sequence evaluation was 7% faster in the back-to-back benchmark.
4. Uncached generation was 2.7% slower; KV caching is needed to test GQA's intended inference
   advantage.
5. Training and inference memory decreased slightly, and the full checkpoint shrank by 37.8 MB.
6. The model remained stable in BF16 while sharing the GPU with Ollama.

## Limitations

- Only one seed was trained.
- The explicit K/V expansion prevents full GQA activation-memory benefits.
- Speed measurements on a desktop GPU can vary with Ollama and display activity.
- Generation uses no KV cache and one qualitative prompt.
- The 0.33% loss regression may vary across repeated runs.

## Conclusion

GQA is a favorable efficiency trade for this model: it removes 10.77% of parameters and improves
full-sequence throughput while increasing validation loss by only 0.0060. Its main expected benefit—
a 75% smaller K/V cache—will be measured after cache-aware generation is implemented.
