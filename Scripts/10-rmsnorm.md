---
type: script
number: 10
topic: rmsnorm
status: done
files: ["10_rms_Norm.py"]
---
## The click
RMS is scale-invariant like LN but NOT shift-invariant (norm(x+5): RMS 4.50/0.980 vs LN 0/0) —
mean-subtraction is wasted compute on transformer activations. eps keeps fp16 alive.
## Acceptance
sec5 dL/dg check ~1e-5 (was 2.16e+01 — the CHECKER was the bug: .sum(0) → .sum((0,1))). [[error-log]]
## Concepts taught
[[rmsnorm]]
## Experiments
