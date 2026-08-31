---
type: concept
---
# RMSNorm
**One line:** RMS=1 per token, no mean-subtraction, 1 param/channel, eps + fp32 upcast for fp16.
**The click:** scale-invariance ≠ shift-invariance; LN's mean-subtraction is dead weight here.
**Learned in:** [[10-rmsnorm]]
**In the GPT:** 3x per block; fp32 reduction in production kernels.
