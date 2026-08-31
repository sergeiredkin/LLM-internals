---
type: script
number: 2
topic: adamw + schedule
status: todo
files: ["02_adamw.py"]
---
## The click
Adam update ~ lr·m/sqrt(v) per coordinate: high lr = oscillation plateau, huge eps = stall,
schedule = a scalar multiplier you can plot.
## TRY checklist
- [ ] lr=2.0 → plateau ~200 (the classic real-world explosion)
- [ ] eps=1e-1 → stalls 2-3 decades early
- [ ] betas=(0.9,0.5) → jittery loss
- [ ] schedule=inverse_sqrt (original Transformer)
## Concepts taught
[[adamw-lr-schedule]] · real-run link: loss spikes → grad clipping
## Experiments
