---
type: project
status: done
---
# Stage 6 — LoRA and QLoRA

## Goal

Train a small task-specific adapter while freezing the pretrained model, then combine the same
low-rank idea with the frozen packed-INT4 base.

## LoRA implementation

For a linear layer with frozen weight `W`, LoRA adds:

`y = x W^T + (alpha / r) x A^T B^T`

`A` is initialized with Kaiming values and `B` is initialized to zero. Therefore the adapter starts
as an exact no-op. Only A and B receive gradients.

The implementation supports:

- target module selection (`q_proj`, `v_proj` by default)
- adapter-only state dictionaries
- strict adapter loading
- exact floating-point merge back into `nn.Linear`
- base-checkpoint SHA-256 protection in generation

## QLoRA connection

`Int4Linear` can be used as the frozen base of `LoRALinear`. The INT4 weights and scales remain
buffers; only A and B are trainable. This is the educational QLoRA path. The current base still
uses unpack/dequantize then floating-point `F.linear`, so it demonstrates memory and optimization
semantics rather than a fused QLoRA kernel.

## Tests

- Zero-initialized adapters preserve base output exactly.
- Base parameters remain frozen and receive no gradients.
- Learned delta rank cannot exceed `r`.
- Merged and unmerged floating-point outputs match within `1.3e-5` maximum logit difference.
- Adapter-only state dictionaries round-trip strictly.
- QLoRA wraps packed INT4 layers without changing initial output.
- All 94 repository tests pass.

## Fixed-batch proofs

Using the 26M RoPE + SwiGLU + GQA checkpoint and one fixed TinyStories batch:

| Mode | Base | Adapter modules | Trainable parameters | Loss |
|---|---|---:|---:|---:|
| LoRA | BF16/floating | 16 Q/V projections | 106,496 | 1.7428 -> 0.0010 |
| QLoRA | packed INT4 group-64 | 16 Q/V projections | 106,496 | 1.7462 -> 0.0010 |

Both proofs used rank 8, alpha 16, learning rate 0.003, 300 steps, batch size 4, context 128,
and seed 2025. The frozen base tensor remained unchanged. Each adapter is about 0.42 MiB.

## Usage

Ordinary LoRA:

```bash
python -m scripts.generate \
  --checkpoint runs/tinystories-rope-swiglu-gqa/best.pt \
  --adapter runs/lora-overfit/adapter-r8.pt \
  --prompt 'Once upon a time' --max-new-tokens 100 \
  --temperature 0.8 --top-k 40 --top-p 0.9 --device cuda --use-kv-cache
```

QLoRA:

```bash
python -m scripts.generate \
  --checkpoint runs/tinystories-rope-swiglu-gqa/best.pt \
  --adapter runs/qlora-overfit/adapter-r8-g64.pt \
  --prompt 'Once upon a time' --max-new-tokens 100 \
  --temperature 0.8 --top-k 40 --top-p 0.9 --device cuda --use-kv-cache
```

The adapter loader verifies that the adapter's recorded base-checkpoint SHA-256 matches the selected
base file.

## Limitation

These adapters were trained only to prove the mechanism by memorizing one batch. They are not a
petroleum-domain adapter and should not be treated as a useful fine-tuned model. The next stage is
to prepare a small licensed petroleum corpus and train/evaluate LoRA or QLoRA with held-out domain
validation.
