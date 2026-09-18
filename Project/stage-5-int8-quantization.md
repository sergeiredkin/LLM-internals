---
type: project
status: done
---
# Stage 5 — Per-Channel INT8 Quantization

## Goal

Learn post-training weight quantization with the smallest auditable implementation before moving
to packed INT4 and QLoRA.

## Method

For each output row of a linear weight matrix:

`scale_i = max(abs(W_i)) / 127`

`Q_i = clamp(round(W_i / scale_i), -127, 127)`

Inference reconstructs `W_hat_i = Q_i * scale_i`. A separate scale per output channel prevents one
row's outlier from reducing precision in every other row.

The current educational kernel dequantizes weights and calls floating-point `F.linear`. It reduces
persistent and serialized storage, but it is not an optimized integer GEMM.

## Scope decision

- Quantize all 56 transformer linear layers: Q/K/V/output projections and three SwiGLU matrices in
  each of eight blocks.
- Preserve the shared token embedding / LM-head matrix in BF16.
- Preserve RMSNorm weights in BF16.
- Keep quantization inference-only and checkpoint-compatible.

Quantizing `lm_head` independently would break weight tying and duplicate the vocabulary matrix.
A later implementation can quantize the shared matrix as one object.

## Correctness checklist

- [x] Per-channel shapes, scale calculation, range, and zero rows.
- [x] Weight reconstruction relative error below 1% on random matrices.
- [x] Quantized linear output relative error below 2%.
- [x] Recursive replacement with qualified-name exclusions.
- [x] Tied embedding and LM head remain tied.
- [x] Quantized state dictionaries round-trip strictly.
- [x] Real checkpoint evaluates over the fixed 409,600-token window.
- [x] Portable format-version-3 export loads and generates on CUDA.

## Result

- Fixed validation loss: `1.837627 -> 1.837698` (`+0.000071`, about `+0.004%`).
- Perplexity: `6.28161 -> 6.28206`.
- FP32 persistent tensor storage during the controlled evaluation: `99.4 -> 36.7 MiB`.
- BF16/INT8 export: `29.5 MiB`, versus `50.3 MiB` for the BF16 export (`41.3%` smaller).
- Peak inference VRAM in the fixed evaluation process: `0.600 -> 0.498 GiB`.
- All 81 repository tests pass.

Export:

`exports/tinystories-26m-int8-hidden.pt`

SHA-256:

`fac49a583f5b2c46f192878e10d71a00151ec67665766cf77919751369f86737`

The export remains local and ignored by Git until a release upload is explicitly approved.

## Interpretation

INT8 introduces almost no aggregate validation degradation. Samples do not match token-for-token
because small logit changes cascade under stochastic decoding. The current dequantize-then-matmul
implementation should not be presented as an INT8 speed optimization; native quantized kernels are
required for that claim.

## Next

Packed groupwise INT4 is complete in [[stage-5-int4-quantization]]. Next, implement ordinary LoRA
before combining adapters with the frozen INT4 base for QLoRA.
