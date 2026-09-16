---
type: project
status: done
---
# Stage 2 — TinyStories

## Goal
Move from character prediction to practical subword tokenization and train a token-level
language model that produces coherent short stories.

## Dataset and tokenizer milestone — 2026-09-16

- Dataset: `roneneldan/TinyStories`
- License: CDLA-Sharing-1.0
- Training subset: 50,000 documents from the first official train shard
- Validation subset: 5,000 documents from the official validation shard
- Split strategy: separate source documents; no random token-level leakage
- Tokenizer: byte-level BPE
- Vocabulary: 8,000, including PAD/UNK/BOS/EOS
- Training tokens: 10,970,105
- Validation tokens: 988,909
- Storage: document JSONL plus `uint16` memory-mapped token streams

Example tokenization:

```text
Once upon a time, a little girl found a shiny red stone near the river.
```

becomes 17 tokens from 71 characters, or **4.18 characters/token**. Frequent words such
as `Once`, `little`, `girl`, and `river` are represented as individual tokens.

## Planned model

- 8 layers
- `d_model=512`
- 8 attention heads
- Context length 512
- 8,000-token vocabulary
- 29,532,672 parameters

The untrained loss is 9.11, close to the expected `ln(8000) = 8.99`.

## GPU smoke test — 2026-09-16

A 20-step BF16 smoke run of the full 29.5M-parameter model completed in 10.94 seconds.

- Validation loss: `9.0730 -> 6.7230` by step 10
- Typical training throughput: approximately 29k tokens/second
- Peak PyTorch VRAM: 1.10 GiB allocated / 1.12 GiB reserved
- Smoke checkpoint: `runs/tinystories-smoke/latest.pt`

Generation already learned spaces, punctuation, and frequent words, but remains incoherent after
only 20 updates, as expected.

A second benchmark increased the micro-batch from 4 to 32 and removed gradient accumulation:

- Throughput: approximately 60k tokens/second (about 2x faster)
- Peak PyTorch VRAM: 5.08 GiB allocated / 5.50 GiB reserved
- Full-run estimate: approximately 25–35 minutes

The fast configuration leaves enough memory for the desktop but cannot coexist safely with a
6.5 GiB Ollama model. Pause the classifier/Ollama workload during fast training.

## Full shared-GPU result — 2026-09-16

The low-memory profile trained all 5,000 steps while the Ollama classifier remained loaded.

- Runtime: **33 minutes 7 seconds**
- Typical throughput: 40k–43k tokens/second
- Peak PyTorch VRAM: **1.12 GiB reserved**
- Initial validation loss: 9.0673
- Best validation loss: **1.8759 at step 4,250**
- Best checkpoint: `runs/tinystories/best.pt`
- Final checkpoint: `runs/tinystories/latest.pt`

The model now generates coherent multi-paragraph children’s stories with characters, dialogue,
simple causality, and conclusions. Generation stops at the learned EOS document boundary.
A deterministic 409,600-token evaluation measured loss **1.9026** and perplexity **6.70**;
the controlled baseline report is `reports/tinystories-baseline.md`.

## Checklist

- [x] Verify official dataset source and license
- [x] Download and checksum official Parquet shards
- [x] Preserve document boundaries in JSONL
- [x] Use the official validation split
- [x] Train an 8K byte-level BPE tokenizer
- [x] Add BOS/EOS document boundaries
- [x] Build memory-mapped token streams
- [x] Add tokenizer inspection tool
- [x] Run model/data integration forward pass
- [x] Run one-batch overfit test with BPE data (`9.0222 -> 0.0000`)
- [x] Run full-model GPU smoke test
- [x] Train and select best validation checkpoint (`val_loss=1.8759`, step 4,250)
- [x] Generate coherent short stories
