# TinyStories Packed Groupwise INT4 Quantization

**Date:** 2026-09-18

## Setup

- Source: 26M RoPE + SwiGLU + GQA checkpoint, step 4,750
- Quantized: 56 transformer linear weights
- Preserved in BF16: tied embedding/LM head and RMSNorm weights
- Quantizer: symmetric packed INT4, group size 64, FP32 scales
- Validation: fixed seed 2025 over 409,600 tokens
- Retraining/calibration: none

## Quality and memory

| Metric | Reference | INT8 hidden | INT4 hidden g64 |
|---|---:|---:|---:|
| Validation loss | 1.837627 | 1.837698 | 1.848032 |
| Perplexity | 6.28161 | 6.28206 | 6.34732 |
| Export size | 50.3 MiB | 29.5 MiB | **20.3 MiB** |
| Peak inference VRAM | 0.600 GiB | 0.498 GiB | **0.488 GiB** |

INT4 reduces file size by 59.6% from BF16 and by 31.2% from hidden INT8. Its validation-loss
increase is 0.010405, or 0.57% relative to the unquantized loss.

## Group-size ablation

| Group size | Validation loss | Perplexity | Persistent storage with FP32 remainder |
|---:|---:|---:|---:|
| 32 | **1.846103** | **6.33509** | 28.8 MiB |
| 64 | 1.848032 | 6.34732 | 27.5 MiB |
| 128 | 1.849963 | 6.35959 | **26.9 MiB** |

Group 64 is the chosen compromise. Group 32 improves loss by 0.00193 but requires approximately
1.2 MiB more persistent storage in the controlled FP32-remainder model. Group 128 saves only about
0.7 MiB and worsens loss by another 0.00193.

The 32 and 128 evaluations ran concurrently on the same GPU to save wall time. Their deterministic
losses are valid, but their throughput measurements were discarded.

## Packing details

- Two signed four-bit values share one byte.
- Packing and unpacking are exactly reversible for integers in `[-8, 7]`.
- Quantization uses symmetric `[-7, 7]` so positive and negative magnitudes have equal range.
- Every row has independent group scales.
- Non-divisible widths are zero-padded before packing and cropped after reconstruction.
- At group size 64, FP32 scales add 0.5 bits per unpadded weight before metadata/padding overhead.

## Performance caveat

This implementation unpacks and dequantizes the complete matrix on every invocation, then calls
floating-point `F.linear`. Generation is therefore slower than the BF16 model. Real INT4 speedups
require fused kernels that consume packed values directly. The result demonstrates compression,
error, and model-quality tradeoffs—not accelerated arithmetic.

## Reproduction

```bash
python -m scripts.evaluate \
  --checkpoint runs/tinystories-rope-swiglu-gqa/best.pt \
  --output reports/results/tinystories-int4-hidden-g64.json \
  --batches 100 --batch-size 8 --evaluation-seed 2025 \
  --device cuda --int4-hidden --int4-group-size 64

python -m scripts.export_int4 \
  --checkpoint runs/tinystories-rope-swiglu-gqa/best.pt \
  --output exports/tinystories-26m-int4-hidden-g64.pt \
  --group-size 64
```

Export SHA-256:

`4d77a944ba95133f4e92a78b4f29957902c79fb04375fb4e0217420c7a7e4a04`

## Decision

Keep group-64 packed INT4 as the frozen-base format for the future educational QLoRA experiment.
Implement and validate ordinary LoRA before combining the two concepts.
