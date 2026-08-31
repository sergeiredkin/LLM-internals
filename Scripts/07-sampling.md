---
type: script
number: 7
topic: sampling
status: todo
files: ["07_sampling.py"]
---
## The click
temp / top-k / top-p / rep-penalty are all edits to ONE logits vector before a softmax;
verbose output shows the mass sliced at every step.
## TRY checklist
- [ ] greedy=True → "the cat sat on the mat" loops forever
- [ ] rep_penalty=2.0 → loops break but text gets weird
- [ ] temperature=1.5 + top_p=0.5 → "creative but not insane"
- [ ] implement min_p yourself (10 lines, same pattern)
## Concepts taught
[[sampling]] · ahead of nanoGPT (only temp+top-k there)
## Experiments
