---
type: project
status: done
---
# Keep/Cut Origin
Founding table of the whole project. Keeps = internals PyTorch doesn't save you from.
Cuts = one-liners PyTorch provides (that's WHY they were cut). The keeps ARE a GPT decomposition.

| Keep | Breaks via | Script | | Cut | Provided by |
|---|---|---|---|---|---|
| attention shapes | OOM, silent mask bugs | 01 | | GEMM | torch matmul |
| AdamW schedule | explodes/plateaus | 02 | | residual | x = x + f(x) |
| KV-cache | long-context OOM | 03 | | dropout | p=0.1, forget |
| LoRA/QLoRA | 7B won't fit | 04 | | | |
| FlashAttention | too slow | 05 | | | |
| quantization | VRAM | 06 | | | |
| sampling | garbage output | 07 | | | |
| cross-entropy | unreadable curves | 08 | | | |
| backprop | gradient bugs | 09 | | | |
| RMSNorm | instability | 10 | | | |

Corollary: the keep list = the debugging manual for the first real run.
