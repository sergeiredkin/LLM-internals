---
type: script
number: 4
topic: lora / qlora
status: todo
files: ["04_lora.py"]
---
## The click
dW = (alpha/r)·B·A has rank <= r — the SVD proves it. LoRA works iff the needed update is
~low-rank; B=0 init is the only safe init; NF4 base stays frozen underneath.
## TRY checklist
- [ ] target_rank=64 with r=16 → MSE plateaus ABOVE full FT (the whole bet, visible)
- [ ] alpha=r + doubled lr → nearly identical curve (alpha/r is part of the LR)
- [ ] QLoRA group_size sweep → NF4 rel-err and MSE move together
## Concepts taught
[[lora-qlora]]
## Experiments
