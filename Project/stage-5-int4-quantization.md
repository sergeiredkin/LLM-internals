---
type: project
status: done
---
# Stage 5 — Packed Groupwise INT4 Quantization

## Goal

Extend the validated INT8 lesson to true four-bit storage while making packing, padding, group
scales, and quality/storage tradeoffs explicit.

## Representation

Weights are quantized symmetrically to integers in `[-7, 7]`. Each signed value is represented by
its four-bit two's-complement nibble, and two nibbles are packed into one `uint8` byte.

For each row and group of 64 input columns:

`scale = max(abs(group)) / 7`

`q = clamp(round(group / scale), -7, 7)`

At group size 64, packed values cost 4 bits/weight and one FP32 scale adds approximately 0.5
bits/weight. SwiGLU's 1360-wide down projection is padded to 1408 columns for grouping, then cropped
back after dequantization.

## Correctness checklist

- [x] Exact signed nibble packing and unpacking, including odd value counts.
- [x] Symmetric quantization range and zero-group behavior.
- [x] Width padding and cropping for non-divisible dimensions.
- [x] Groupwise reconstruction and linear-output error tests.
- [x] Strict INT4 state-dictionary round trip.
- [x] GPT integration while preserving the tied embedding/LM head.
- [x] Fixed 409,600-token validation.
- [x] Group-size 32/64/128 quality and storage sweep.
- [x] Portable format-version-4 export and CUDA generation.

## Group-size sweep

| Group | Fixed loss | Change from 1.837627 | FP32 remainder + INT4 storage |
|---:|---:|---:|---:|
| 32 | 1.846103 | +0.008476 | 28.8 MiB |
| 64 | 1.848032 | +0.010405 | 27.5 MiB |
| 128 | 1.849963 | +0.012336 | 26.9 MiB |

Group 64 is retained as the balance point. Smaller groups reduce outlier impact but require more
scales; larger groups save little additional space while degrading quality further.

## Final result

- BF16 export: 50.3 MiB.
- Hidden INT8 export: 29.5 MiB.
- Hidden INT4 group-64 export: **20.3 MiB**.
- INT4 is 59.6% smaller than BF16 and 31.2% smaller than the INT8 export.
- Fixed loss increases by 0.010405, or 0.57%.
- Peak inference VRAM in the fixed run decreases from 0.600 to 0.488 GiB.

Export:

`exports/tinystories-26m-int4-hidden-g64.pt`

SHA-256:

`4d77a944ba95133f4e92a78b4f29957902c79fb04375fb4e0217420c7a7e4a04`

The binary remains local and Git-ignored until release upload approval.

## Kernel limitation

`Int4Linear` unpacks and dequantizes on every call before floating-point `F.linear`. It proves
storage and quality behavior but is slower than a fused native INT4 kernel. Packed storage alone
does not imply faster arithmetic.

## Next

Build LoRA on ordinary floating-point linear layers first. Then freeze this validated INT4 base and
train only LoRA adapters to form the educational QLoRA path.
