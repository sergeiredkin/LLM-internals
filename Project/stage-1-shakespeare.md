---
type: project
status: done
---
# Stage 1 — Shakespeare

## Goal
Assemble and train a complete character-level decoder-only GPT, reaching approximately
1.5 validation loss on Tiny Shakespeare.

## Result
Completed on 2026-09-16 with an NVIDIA RTX 3060 12 GB while sharing the GPU with the
local classifier service.

- Model: 6 layers, `d_model=384`, 6 heads, context 256
- Parameters: 10,745,088
- Tokenizer: 65 characters
- Precision: BF16
- Effective batch: 64 sequences / 16,384 tokens per update
- Training steps: 5,000
- Best checkpoint: step 2,250
- Best validation loss: **1.4910**
- Peak PyTorch VRAM: **0.45 GiB allocated / 0.47 GiB reserved**
- Typical throughput: 75k–83k tokens/second
- Checkpoint: `runs/shakespeare/best.pt`

Later steps reduced training loss but worsened validation loss, demonstrating overfitting.
Generation should therefore use `best.pt`, not `latest.pt`.

## Checklist
- [x] Download and verify Tiny Shakespeare
- [x] Build deterministic character tokenizer
- [x] Build memory-mapped train/validation data pipeline
- [x] Implement causal GPT with SDPA, RMSNorm, and tied embeddings
- [x] Verify causal masking and finite gradients
- [x] Overfit one fixed batch
- [x] Run BF16 GPU smoke test
- [x] Train full model with checkpoints and validation
- [x] Reach the target validation loss
- [x] Generate coherent Shakespeare-like dialogue

## Example

```text
ROMEO:
What is the world, when the sea miserable,
Or that rude the gates such at him with his bard,
As was, away, when he dares him hence,
Let forth him that now incannot weep.

AUTOLYCUS:
Alack the news in his peace, sir.
```
