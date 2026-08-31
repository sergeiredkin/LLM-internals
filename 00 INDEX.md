---
type: moc
updated:
---
# LLM GPT Trial — Vault Index

**What this is:** learn LLM internals by running instrumented scripts, then build a from-scratch GPT.
**Origin:** keep/cut triage — [[keep-cut-origin]].

## Start here
- [[01 Conventions]] — how this vault works, how to add notes
- [[playbook]] — what to do when anything breaks
- [[failure-modes]] — the 5 ways things break

## Scripts
| # | Note | Topic | Status | Key artifact |
|---|---|---|---|---|
| 01 | [[01-attention-shapes]] | shapes, masks, OOM math | todo | — |
| 02 | [[02-adamw-schedule]] | LR schedules, divergence | todo | plot |
| 03 | [[03-kv-cache]] | decode complexity, GQA | done | counters |
| 04 | [[04-lora-qlora]] | rank, alpha, NF4 base | todo | MSE table |
| 05 | [[05-flash-attention]] | online softmax tiling | todo | — |
| 06 | [[06-quantization]] | int4/NF4, outliers | done | 5 PNGs |
| 07 | [[07-sampling]] | temp/top-k/top-p | todo | — |
| 08 | [[08-cross-entropy]] | stability, gradient, LS floor | todo | — |
| 09 | [[09-backprop]] | manual vs autograd, residuals | done | 09_grad_norms.png |
| 10 | [[10-rmsnorm]] | RMS vs LN, eps, fp16 | done | — |

Suggested order: 01 → 08 → 09 → 10 → 02 → 07 → 03 → 04 → 06 → 05

## Build a GPT
- [[GPT-roadmap]] — assembly map + 4 stages
- [[stage-1-shakespeare]] · [[stage-2-tinystories]] · [[stage-3-modern-pass]] · [[stage-4-gpt2-124m]]

## Troubleshooting
[[failure-modes]] · [[playbook]] · [[error-log]] · [[toolbox]] · [[gotchas]] · [[expected-magnitudes]] · [[baselines]]

## Concepts
[[attention]] · [[adamw-lr-schedule]] · [[kv-cache]] · [[lora-qlora]] · [[flash-attention]] · [[quantization]] · [[sampling]] · [[cross-entropy]] · [[backprop]] · [[rmsnorm]]
Stubs: [[rope]] · [[tokenizer]] · [[mixed-precision]] · [[distributed-training]]
