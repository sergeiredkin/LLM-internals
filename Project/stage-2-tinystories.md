---
type: project
status: doing
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
- [ ] Run full-model GPU smoke test
- [ ] Train and select best validation checkpoint
- [ ] Generate coherent short stories
