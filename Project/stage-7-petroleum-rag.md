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

A separate table-candidate extraction path is now available. It detects table-like pages, preserves their provenance, and writes independently chunked records without mixing them into prose retrieval:

```bash
python -m scripts.extract_petroleum_tables \
  --pages data/petroleum/pages-expanded.jsonl \
  --output data/petroleum/table-candidates.jsonl \
  --merge-continuations
```

The current threshold produces 97 candidate table groups. Continuation pages are merged when a
labelled continuation is detected; for example, the NPR-4 Table 4 group preserves pages 55–57.
A row-candidate path is also available:

```bash
python -m scripts.reconstruct_petroleum_tables \
  --pages data/petroleum/pages-expanded.jsonl \
  --output data/petroleum/table-row-candidates.jsonl
```

It produces 216 auditable row-like records. It deliberately does not fabricate column alignment;
A conservative fact-candidate path is also available:

```bash
python -m scripts.construct_petroleum_table_facts \
  --rows data/petroleum/table-row-candidates.jsonl \
  --output data/petroleum/table-fact-candidates.jsonl
```

It emits 128 fact candidates with explicit confidence and evidence text. It extracts facts such as
“dry gas” and “south of Barrow” but does not claim that PDF column alignment is fully recovered.

Section-aware ranking is available with `--section-aware`. It applies conservative petroleum intent
signals for salt/seals, reservoir/facies, source beds, gas hydrates, margins, aquifers, and summary
passages. On the 50-question benchmark, augmented retrieval with section-aware ranking and primary
reranking reached hit@5 0.980, recall@5 0.962, and MRR@5 0.708.

Summary-aware ranking is available with `--summary-aware` and boosts query-matching abstract,
summary, and introduction openings. The lightweight semantic reranker is available with
`--semantic-rerank`; it scores query-term coverage across the passage, title, and opening context.
On the 50-question benchmark, the combined configuration reached hit@5 0.980, recall@5 0.981,
and MRR@5 0.759.

The optional augmented corpus can be built with:

```bash
python -m scripts.build_petroleum_augmented \
  --base data/petroleum/chunks-expanded-clean.jsonl \
  --table-facts data/petroleum/table-fact-candidates.jsonl \
  --output data/petroleum/chunks-expanded-with-table-facts.jsonl
```

On the 18-question benchmark, the augmented corpus improved hit rate@5 from 0.889 to 0.944 and
recall@5 from 0.850 to 0.900, while MRR moved from 0.592 to 0.578. An optional primary-evidence rerank is available:

```bash
python -m scripts.evaluate_retrieval ... --prefer-primary-evidence
```

With the current conservative 0.90 downweight for derived facts, MRR returns to 0.592 and the
benchmark remains at the original 0.889 hit rate / 0.850 recall. This gives us an explicit choice:
use augmented retrieval for recall, or primary-evidence reranking for answer quality.

Answer-level evaluation is now available through `scripts/evaluate_answers.py`. Each JSONL record
contains `query`, `answer`, `citations`, and `evidence` records with `document_id` and `text`.
The evaluator reports citation coverage, lexical support, sentence-level support, numeric/unit grounding,
abstention accuracy, unsupported-sentence rate, and a conservative grounded-answer rate. It is deliberately separate from retrieval
metrics so a retrieved passage is not mistaken for a supported answer. The draft generator
`scripts/build_answer_eval_draft.py` creates provenance-linked records for manual rewriting; its
extractive smoke-test scores must not be treated as final answer quality.

The retrieval-only CLI is:

```bash
python -m scripts.retrieve_context \
  --documents data/petroleum/chunks-expanded-clean.jsonl \
  --pre-chunked \
  --query "What controls bottom-hole pressure?" \
  --top-k 5
```

It returns numbered source citations and exits with status 2 when it abstains. The output should be
passed to an answer model only after checking the citations.

The evaluation CLI is:

```bash
python -m scripts.evaluate_retrieval \
  --documents data/petroleum/chunks-expanded-clean.jsonl \
  --pre-chunked \
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

`reports/petroleum-expansion-pilot.md` records the original expansion to 20 checksum-verified USGS
reports. The current corpus also includes a separate 34-page Expedition Summary source for the
Indian National Gas Hydrate Program: 21 manifest sources, 1,869 recovered pages, 4,818 chunks,
4,275 clean chunks, and 543 rejected chunks. The fixed 18-question evaluation is now run with
parent-page grouping and optional controlled query expansion: hit rate@5 0.833, recall@5 0.842,
and MRR@5 0.536. Do not expand toward 50 reports until the remaining misses are resolved.

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

- [x] Select and document the 20-report USGS pilot corpus and record checksums.
- [x] Add a checksum-verified Expedition Summary source for the Indian Gas Hydrate question.
- [x] Add manifest, checksum, PDF extraction, chunking, and review-sample pipeline.
- [x] Add conservative filtering for figure-only, short, bibliography, and corrupted chunks.
- [x] Create and manually confirm an auditable label file for the 50 pilot review records.
- [x] Recover the malformed PDF page as a provenance-preserving figure-only record.
- [ ] Resolve remaining table and extraction defects; table candidates are now isolated for follow-up.
- [x] Normalize the selected corpus to JSONL and create a deterministic fixed query set.
- [ ] Create deterministic train/validation splits for future adaptation.
- [x] Establish BM25 metrics on the fixed query set: hit rate@5 0.800, recall@5 0.818, MRR@5 0.683 on the expanded corpus.
- [x] Add context assembly with citations, page/source IDs, and an abstention rule.
- [x] Add EIA normalized local storage, ingestion, read-only queries, and calculations.
- [x] Add a strict BSEE production importer with provenance and fail-closed schema validation.
- [ ] Create petroleum instruction examples only from retrieved, licensed evidence.
- [ ] Train and evaluate a small QLoRA adapter.
