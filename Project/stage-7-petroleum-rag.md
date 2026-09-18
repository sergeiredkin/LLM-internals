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

## Current pilot result

`reports/petroleum-pilot-week1.md` records 12 reports, 1,461 extracted/recovered pages, 3,304 chunks,
and one recovered figure-only page. The first filter retains 2,859 chunks and rejects 445 obvious
extraction failures. The 50-sample review labels 30 as acceptable for prose retrieval and 20 as
rejected or reserved for table/figure processing. The extraction pipeline is working but the corpus
is not yet production-ready.

## Corpus expansion pilot

`reports/petroleum-expansion-pilot.md` records expansion from 12 to 20 checksum-verified USGS
reports: 1,838 recovered pages, 4,613 chunks, 4,069 clean chunks, and 544 rejected chunks. The
fixed BM25 metrics remain hit rate@5 0.800, recall@5 0.818, and MRR@5 0.683. A 100-sample review
must be completed before expanding toward 50 reports.

## Week 2 EIA pilot

`reports/eia-week2-pilot.md` documents the live EIA pull. SQLite is used locally because PostgreSQL
is not installed; the schema is relational and ready for migration. Crude production, inventories,
refinery inputs, and spot prices were ingested into the pilot store. Product supplied was temporarily
rate-limited by the EIA API and must be retried with a registered key.

Example:

```bash
python -m scripts.query_eia \
  --database data/petroleum/eia.sqlite \
  --series MCRFPUS1 \
  --yoy-period 2025-01
```

## Week 3 BSEE pilot

`reports/bsee-week3-pilot.md` documents the constrained BSEE layer. Official ranked operator exports
were found, but their aggregate fixed-width columns are not yet safe to interpret automatically. The
importer therefore requires an explicit mapped CSV with oil, gas, and water units plus source URL and
checksum. No ambiguous BSEE values are loaded.

## Next steps

- [x] Select and document a 20-report USGS pilot corpus and record checksums.
- [x] Add manifest, checksum, PDF extraction, chunking, and review-sample pipeline.
- [x] Add conservative filtering for figure-only, short, bibliography, and corrupted chunks.
- [x] Create and manually confirm an auditable label file for the 50 pilot review records.
- [x] Recover the malformed PDF page as a provenance-preserving figure-only record.
- [ ] Resolve remaining table and extraction defects.
- [x] Normalize the selected corpus to JSONL and create a deterministic fixed query set.
- [ ] Create deterministic train/validation splits for future adaptation.
- [x] Establish BM25 metrics on the fixed query set: hit rate@5 0.800, recall@5 0.818, MRR@5 0.683 on the expanded corpus.
- [x] Add context assembly with citations, page/source IDs, and an abstention rule.
- [x] Add EIA normalized local storage, ingestion, read-only queries, and calculations.
- [x] Add a strict BSEE production importer with provenance and fail-closed schema validation.
- [ ] Create petroleum instruction examples only from retrieved, licensed evidence.
- [ ] Train and evaluate a small QLoRA adapter.
