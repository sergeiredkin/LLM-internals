---
type: project
status: doing
---
# Stage 3 — SwiGLU Controlled Ablation

## Question

Does replacing the GELU MLP with a parameter-matched SwiGLU improve the RoPE TinyStories model
when all other settings remain fixed?

## Mental model

The baseline MLP has one expanded stream:

```text
down(GELU(up(x)))
```

SwiGLU has a value stream and a learned gate:

```text
down(SiLU(gate(x)) * value(x))
```

The elementwise multiplication lets the gate decide which expanded features pass through.

## Fair parameter matching

A bias-free GELU MLP has two weight matrices; SwiGLU has three. Keeping hidden width 2,048 would
therefore give SwiGLU substantially more parameters. The fair target is two thirds of the GELU
width:

```text
2/3 × 2048 = 1365.33
```

We use 1,360, the nearest multiple of 16. Per transformer block:

| MLP | Hidden width | Weight parameters |
|---|---:|---:|
| GELU | 2,048 | 2,097,152 |
| SwiGLU | 1,360 | 2,088,960 |

SwiGLU has 8,192 fewer parameters per block, a difference of 0.39% within the MLP. Across eight
blocks, the planned model will have 65,536 fewer parameters than the RoPE control.

## Checklist

### Isolated component

- [x] Implement `SwiGLU` independently from GPT.
- [x] Implement hardware-friendly matched-width calculation.
- [x] Verify the exact `down(SiLU(gate(x)) * value(x))` formula.
- [x] Verify input/output shapes.
- [x] Verify BF16 behavior and finite gradients.
- [x] Verify parameter count differs from GELU by less than 0.5%.
- [x] Preserve the existing GELU `MLP` unchanged.

### Integration

- [x] Add validated `mlp_type: gelu | swiglu` configuration.
- [x] Select the MLP implementation inside each transformer block.
- [x] Keep GELU as the default for old configuration/checkpoint compatibility.
- [x] Test forward, backward, causality, generation, and parameter count in both modes.

### Training gates

- [x] Pass one-batch overfitting (`8.9948 -> 0.0000` in 300 steps).
- [x] Pass a 20-step full-size BF16 GPU smoke test.
- [ ] Create a controlled 5,000-step RoPE + SwiGLU configuration.
- [ ] Train and select the best deterministic-validation checkpoint.

### Evaluation and publication

- [ ] Evaluate the same fixed 409,600 validation tokens with seed 2025.
- [ ] Generate with the same prompt and seed 42.
- [ ] Compare against RoPE loss 1.8401 and perplexity 6.30.
- [ ] Report parameters, speed, VRAM, loss, perplexity, and sample quality.
- [ ] Commit and push code, tests, config, results, and report.

## Log

- 2026-09-17: isolated SwiGLU and matched-width helper implemented. All 37 tests pass. GPT remains
  unchanged.
- 2026-09-17: integrated the validated MLP switch. Both real GELU checkpoints load strictly; a
  full-size 29,204,992-parameter RoPE + SwiGLU model passes BF16 forward/backward. All 41 tests
  pass.
- 2026-09-17: RoPE + SwiGLU one-batch overfit passed on CPU: loss 8.9948 to 0.0000 in
  300 steps.
- 2026-09-17: full-size 20-step BF16 smoke test passed. Training loss fell 9.0892 to 6.3621,
  validation loss fell 9.0497 to 6.8153, typical throughput reached 24k–28k tokens/s, gradients
  remained finite, and peak PyTorch memory was 1.19 GiB reserved.
