---
type: script
number: 8
topic: cross-entropy
status: todo
files: ["08_ce.py"]
---
## The click
CE = log_softmax + pick; gradient IS (softmax − onehot)/B; label smoothing installs a hard
loss FLOOR; init loss = ln(V) or you are worse than random.
## TRY checklist
- [ ] logit_scale=50 → naive log(softmax) = -inf, log_softmax unaffected
- [ ] label_smoothing=0.5 → huge floor; optimal p_true < 1 caps logit gaps
- [ ] B=1 vs 8 → gradient scale is exactly 1/B
## Concepts taught
[[cross-entropy]]
## Experiments
