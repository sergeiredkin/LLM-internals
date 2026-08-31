---
type: concept
---
# Attention
**One line:** softmax(QK^T/sqrt(dk))V over (B,H,T,dk); causal mask = lower-triangular.
**The click:** scores are (B,H,T,T) — the OOM spot; broadcasting lets mask/head bugs run silently.
**Learned in:** [[01-attention-shapes]]
**In the GPT:** every block; ship as F.scaled_dot_product_attention(is_causal=True).
**Open questions:** positions ([[rope]]), QK-norm.
