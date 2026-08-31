---
type: script
number: 1
topic: attention shapes
status: todo
files: ["01_attention.py"]
---
## The click
Everything is (B,H,T,dk); scores are (B,H,T,T) and OOM exactly there. Broadcasting makes
mask/head bugs run SILENTLY — rowsums=1 proves nothing.
## TRY checklist
- [ ] H=5 → view() RuntimeError (read it)
- [ ] cross_attn=True → rectangular causal mask still works
- [ ] mask_bug=True → attends to the FUTURE silently
- [ ] scale_by_sqrt_d=False → softmax saturation = why 1/sqrt(dk) exists
## Concepts taught
[[attention]] · builds toward [[flash-attention]], [[GPT-roadmap]]
## Experiments
