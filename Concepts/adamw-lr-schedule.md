---
type: concept
---
# AdamW + LR schedule
**One line:** m/sqrt(v) ≈ per-coordinate step ~lr; warmup→cosine; spikes → grad clip 1.0.
**The click:** the optimizer, not the data, shapes the loss curve — plot LR(t) next to loss.
**Learned in:** [[02-adamw-schedule]]
**In the GPT:** training loop; betas (0.9, 0.95), wd on 2D params only.
