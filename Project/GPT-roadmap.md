---
type: project
status: doing
---
# GPT Build Roadmap
**Thesis:** keeps + cuts + glue + one new concept ([[rope]]) = nanoGPT-or-better.
Independently derived — Llama-flavored (RMSNorm, RoPE, GQA, SwiGLU, top-p), not a nanoGPT copy.
**Correctness proof = convergence:** matching reference val-loss is not copying; it's math verified.

## Assembly map
| GPT component | From script | Production swap |
|---|---|---|
| causal MHA | [[01-attention-shapes]] | nn.Linear + SDPA(is_causal=True) |
| FlashAttention | [[05-flash-attention]] | ship the SDPA call |
| RMSNorm x3/block | [[10-rmsnorm]] | nn.Module, fp32 reduction |
| MLP / init | [[09-backprop]] | GELU or SwiGLU; N(0,0.02), out-proj /sqrt(2L) |
| CE loss | [[08-cross-entropy]] | F.cross_entropy, logits.view(-1,V) |
| AdamW+schedule | [[02-adamw-schedule]] | betas (0.9,0.95), wd on 2D only |
| KV-cache infer | [[03-kv-cache]] | preallocate (already ahead of nanoGPT) |
| sampling | [[07-sampling]] | full suite |
| LoRA / int4 | [[04-lora-qlora]] / [[06-quantization]] | post-training passes |
| NEW | [[rope]] or learned pos-emb | — |

## Stages
1. [[stage-1-shakespeare]] — 10.7M char-LM, val ~1.5, T4 ~15 min. Proves the assembly.
2. [[stage-2-tinystories]] — 20–50M BPE, overnight, coherent English. "It speaks."
3. [[stage-3-modern-pass]] — RoPE/GQA/SwiGLU + LoRA + int4. Exercises 03/04/06/07.
4. [[stage-4-gpt2-124m]] — FineWeb, ~4 days 8xA100. Code unchanged — that's the point.

## Debugging creed
First real run's failures = the keep table: OOM→01/03 · spike→02+clip · NaN→06/10 fp16 ·
garbage→07 · "flat at 10.8?"→08 ln(V) · silent wrong→01 mask bug. Playbook: [[playbook]].
