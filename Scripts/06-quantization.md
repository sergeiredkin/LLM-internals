---
type: script
number: 6
topic: quantization
status: done
runtime: cpu
deps: [torch]
files: ["06_quantization.py"]
---
## Purpose
Quantize→dequantize at int8/4/2 with tensor/channel/group scales + NF4; measure error.
## The click
Error is dominated by (a) the largest value in the group setting the scale (outliers) and
(b) how many weights share one scale (granularity). NF4 wins by matching level density to N(0,1).
## Key CFG knobs
bits · mode (tensor|channel|group) · group_size · nf4 · outlier_frac · n_threads
## Expected / acceptance
[[baselines]] — bit-identical under seed 0; table numbers MUST equal chart numbers.
## Artifacts
![[weight_distribution.png]]
![[error_distribution.png]]
![[granularity_sweep.png]]
![[bits_sweep.png]]
![[outlier_sweep.png]]
## TRY checklist
- [ ] outlier_frac=0.01, mode=tensor → collapse (why LLM.int8/QLoRA exist)
- [ ] nf4=True vs symmetric int4 @ g=64
- [ ] bits=2 + group=32 (usable) vs bits=2 + tensor (garbage)
- [ ] skewed weights W = W.abs()**2 * W.sign() → asymmetric wins
- [ ] NF4 scan vs broadcast one-liner — timing lesson (bandwidth > flops)
## Concepts taught
[[quantization]] · [[lora-qlora]]
## Incidents (see [[error-log]])
NF4 broadcast ~0.8GB temp · plt.show() block · ghost processes/RAM · chart≠table mismatch ·
matplotlib hist hang (→ ASCII, later matplotlib re-added properly)
## Experiments
- baseline confirmed across 4 runs, bit-identical (seed 0 works as regression test)
- found + fixed: plot function ignored its input, built a fresh 512² matrix (row 13)
