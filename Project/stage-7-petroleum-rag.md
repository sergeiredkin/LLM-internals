---
type: project
status: doing
---
# Stage 7 — Petroleum RAG Foundation

## Goal

Build a retrieval baseline before domain fine-tuning. RAG must first prove that the system can find
the right source passage and preserve provenance. QLoRA comes afterward, using only licensed domain
examples.

## Implemented first milestone

`llm/rag.py` now provides:

- normalized deterministic tokenization
- JSONL document loading and writing
- word-based chunks with configurable overlap
- source, title, document ID, and metadata preservation
- dependency-free Okapi BM25 retrieval
- deterministic score tie-breaking
- hit rate, recall, and MRR evaluation
- citation-labelled context assembly
- explicit abstention when no evidence scores positively

The retrieval-only CLI is:

```bash
python -m scripts.retrieve_context \
  --documents data/petroleum/documents.jsonl \
  --query "What controls bottom-hole pressure?" \
  --top-k 5
```

It returns numbered source citations and exits with status 2 when it abstains. The output should be
passed to an answer model only after checking the citations.

The evaluation CLI is:

```bash
python -m scripts.evaluate_retrieval \
  --documents data/petroleum/documents.jsonl \
  --queries data/petroleum/queries.jsonl \
  --output reports/results/petroleum-retrieval.json
```

No petroleum documents are included yet. `data/petroleum/README.md` defines the license and source
manifest requirements. The accelerated Week-1 pipeline now includes:

```bash
python -m scripts.ingest_petroleum_pdfs \
  --manifest data/petroleum/manifest.jsonl \
  --output data/petroleum/pages.jsonl

python -m scripts.prepare_petroleum_chunks \
  --pages data/petroleum/pages.jsonl \
  --chunks data/petroleum/chunks.jsonl \
  --review data/petroleum/review-50.jsonl
```

PDF pages retain page numbers and source checksums. The review file deliberately samples across the
whole corpus so extraction defects are found before bulk ingestion.

## Evaluation protocol

Create a held-out query JSONL file with this schema:

```json
{"query":"What controls bottom-hole pressure?","relevant_document_ids":["source-001"]}
```

Keep the query set fixed while comparing:

- chunk sizes and overlap
- BM25 versus dense retrieval
- top-k values
- reranking strategies
- RAG answer faithfulness

Report hit rate@k, recall@k, MRR@k, source citation accuracy, and an abstention rate for questions
that are not supported by the corpus.

## Guardrails

- Do not invent petroleum documents or use proprietary material without permission.
- Keep raw and processed data out of Git.
- Store checksums, licenses, and retrieval dates in a manifest.
- Separate retrieval evaluation from language-model generation quality.
- QLoRA training must use train/validation separation and a fixed held-out question set.

## Next steps

- [ ] Select and document a redistributable petroleum corpus.
- [x] Add manifest, checksum, PDF extraction, chunking, and review-sample pipeline.
- [ ] Normalize the selected corpus to JSONL and create deterministic train/validation/query splits.
- [ ] Establish BM25 metrics on the fixed query set.
- [x] Add context assembly with citations and an abstention rule.
- [ ] Create petroleum instruction examples only from retrieved, licensed evidence.
- [ ] Train and evaluate a small QLoRA adapter.
