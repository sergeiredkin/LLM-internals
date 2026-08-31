---
type: script
number: 9
topic: backprop
status: done
files: ["09_back_prop.py"]
---
## The click
Backward = chain rule through stored activations; manual ≡ autograd to 1e-17. Residual needs
CONSTANT width → layer 0 is a stem → why GPT blocks preserve d_model. Residual = flat grad norms.
## Acceptance
sec1 ≤ 1e-12 both residual modes; sec3: no-residual stalls ~0.9–1.0, residual → ~0.2–0.5
(gap widens at depth 32).
## Incidents (3, see [[error-log]])
- matmul dtype strict (double != float) → dtype= in every constructor
- (y-tt)**2 .mean() parsed as (y-tt)**(2.mean()) → parenthesize
- 8-vs-16 residual add → the stem convention
## Run
MPLBACKEND=Agg python3 09_back_prop.py → 09_grad_norms.png
## Concepts taught
[[backprop]]
## Experiments
