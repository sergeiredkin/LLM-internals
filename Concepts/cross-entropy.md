---
type: concept
---
# Cross-entropy
**One line:** grad = (softmax − onehot)/B; loss floor from label smoothing; init loss = ln(V).
**The click:** loss curves are readable diagnostics — ln(V) at init, floor at the end, spikes = LR.
**Learned in:** [[08-cross-entropy]]
**In the GPT:** F.cross_entropy on logits.view(-1, V).
