# TinyStories Per-Channel INT8 Quantization

**Date:** 2026-09-18

## Setup

- Source: 26,059,264-parameter RoPE + SwiGLU + GQA checkpoint, step 4,750
- Quantized: all 56 transformer linear weights
- Preserved: tied embedding/LM head and RMSNorm weights
- Scheme: symmetric INT8, one FP32 scale per output channel
- Validation: seed 2025, fixed 409,600-token window
- No retraining or calibration corpus

## Quality

| Metric | BF16/autocast reference | Hidden INT8 | Change |
|---|---:|---:|---:|
| Validation loss | 1.837627 | 1.837698 | +0.000071 |
| Perplexity | 6.28161 | 6.28206 | +0.00045 |
| Peak inference VRAM | 0.600 GiB | 0.498 GiB | -17.0% |

The loss increase is approximately 0.004%, so per-channel INT8 preserves aggregate model quality on
this checkpoint.

## Storage

| Artifact/storage | Size |
|---|---:|
| FP32 model tensors before quantization | 99.4 MiB |
| FP32 non-linear tensors + INT8 hidden tensors | 36.7 MiB |
| Existing BF16 inference file | 50.3 MiB |
| BF16 + hidden-INT8 inference file | **29.5 MiB** |

The portable file is 41.3% smaller than the BF16 export. It cannot approach a full 50% reduction
because the tied 8,000 × 512 embedding/LM-head matrix remains BF16 and each output channel needs a
scale.

## Performance caveat

`Int8Linear` dequantizes its weight on every forward call and then invokes floating-point
`F.linear`. This implementation teaches representation and error behavior, but does not use an
INT8 GEMM kernel. Timing from separate fixed-evaluation runs is too sensitive to GPU state and
generated EOS length for a strong speed comparison. Native kernels will be a separate optimization.

## Qualitative behavior

The fixed stochastic sample changed names and events despite nearly unchanged validation loss.
That is expected: a tiny quantization perturbation can alter one sampled token, after which every
later conditional distribution sees a different prefix. Sample identity is therefore not a useful
quantization correctness requirement.

## Reproduction

```bash
python -m scripts.evaluate \
  --checkpoint runs/tinystories-rope-swiglu-gqa/best.pt \
  --output reports/results/tinystories-int8-hidden.json \
  --batches 100 --batch-size 8 --evaluation-seed 2025 \
  --generation-seed 42 --device cuda --int8-hidden

python -m scripts.export_int8 \
  --checkpoint runs/tinystories-rope-swiglu-gqa/best.pt \
  --output exports/tinystories-26m-int8-hidden.pt
```

Export SHA-256:

`fac49a583f5b2c46f192878e10d71a00151ec67665766cf77919751369f86737`

## Decision

Keep per-channel INT8 as the validated first quantization baseline. Proceed to packed groupwise
INT4, where scale overhead, nibble packing, outliers, and kernel limitations become more important.
