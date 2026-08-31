---
type: concept
---
# LoRA / QLoRA
**One line:** learn a rank-r delta (alpha/r)·B·A on a frozen (NF4-quantized) base.
**The click:** the low-rank bet — if the task's update is ~rank r, 0.5% of params match full FT.
**Learned in:** [[04-lora-qlora]]
**In the GPT:** stage-3 style fine-tuning; B=0 init is the only safe init.
