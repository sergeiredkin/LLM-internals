---
type: script
number: 5
topic: flash attention
status: todo
files: ["05_flash.py"]
---
## The click
Never materialize T×T: keep running max m, running sum l, rescale the accumulator by
exp(m_old − m_new) every block. verbose_tiny shows the rescale — THAT is FlashAttention.
## Notes
- bench_sdpa=True needs a CUDA runtime (Colab GPU); CPU prints a hint and skips.
- Production = call F.scaled_dot_product_attention; the hand loop is for understanding.
## Concepts taught
[[flash-attention]]
## Experiments
