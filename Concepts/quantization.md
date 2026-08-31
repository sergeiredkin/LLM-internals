---
type: concept
---
# Quantization
**One line:** fp32 → few bits + a scale; error ∝ range/granularity; outliers are the enemy.
**The click:** per-tensor scale = extreme-value statistic (sample-dependent); group scale =
LLN statistic (distribution-robust). That's why production went groupwise.
**Learned in:** [[06-quantization]]
**In the GPT:** stage-3 int4 pass; KV-cache int8/int4.
**Open questions:** activation quantization (harder than weights) · GPTQ/AWQ vs RTN.
