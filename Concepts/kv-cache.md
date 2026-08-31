---
type: concept
---
# KV-cache
**One line:** prefill once, then decode each token against cached K/V — O(T^2) total, not O(T^3).
**The click:** correctness gate = match full re-forward to float tolerance (1e-6..1e-5).
**Learned in:** [[03-kv-cache]]
**In the GPT:** inference; preallocate cache (no torch.cat per step); GQA halves memory.
