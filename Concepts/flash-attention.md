---
type: concept
---
# FlashAttention
**One line:** online-softmax tiling — softmax is a stream; memory O(T), not O(T^2).
**The click:** keep (m, l, acc), rescale per block by exp(m_old − m_new).
**Learned in:** [[05-flash-attention]]
**In the GPT:** ship as the SDPA call the loop explains.
