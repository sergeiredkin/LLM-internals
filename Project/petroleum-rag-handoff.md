# Petroleum RAG Handoff Note

**Date:** 2026-10-05  
**Repository:** `/home/sergei/Documents/learngpt`  
**Branch:** `main`  
**Remote:** `https://github.com/sergeiredkin/LLM-internals.git`

## Mission

Build a reliable petroleum-domain RAG system with provenance, page-level citations, table support, answer grounding, conservative abstention, reproducible evaluation, and eventual licensed fine-tuning only if it is justified.

## Current status

The working tree is clean. The latest pushed commit is:

```text
9e85948 — no-mistakes(review): Remove continue-on-error so CI fails on test failures
```

Full test suite:

```text
125 tests passed
```

The NVIDIA RTX 3060 12 GB is available through CUDA.

## Work completed

### Corpus and provenance

- Built a checksum-tracked petroleum manifest at `data/petroleum/manifest.jsonl`.
- Expanded the verified corpus from 21 to **27 USGS/OSTI sources** (latest: `usgs-of-2013-1094-bakken`, domain-report).
- Preserved official URLs, source IDs, titles, publishers, publication dates, licenses, SHA-256 hashes, and source roles.
- Added `scripts/validate_petroleum_manifest.py` for duplicate-ID, local-file, and checksum validation.
- Added page-aware PDF ingestion through `scripts/ingest_petroleum_pdfs.py`, now with retry/backoff and a longer timeout for slow USGS/OSTI servers.
- Extraction stats as of the 26-source corpus (not yet re-measured for source 27):
  - 26 sources
  - 2,082 recovered pages
  - 4,555 clean chunks
  - 808 rejected chunks
  - 1 extraction failure retained for review
- Older/scanned material is filtered conservatively; corrupted, figure-only, sparse, and too-short chunks are not silently included.

### Retrieval

Production retrieval is lexical-first:

```text
BM25
→ query expansion
→ table-fact augmentation
→ primary-evidence preference
→ source-aware weighting
→ section-aware ranking
→ summary-aware ranking
→ lightweight semantic reranking
```

Implemented in `llm/rag.py` and exposed through the retrieval scripts.

Source roles currently include:

```text
domain-report
methodology
overview
```

Overview sources can be downranked with `--source-aware` so short high-level reports do not outrank direct evidence.

### Tables

Implemented:

- `scripts/extract_petroleum_tables.py`
- `scripts/reconstruct_petroleum_tables.py`
- `scripts/construct_petroleum_table_facts.py`
- parent-page grouping
- table continuation merging
- row reconstruction
- conservative table facts
- provenance preservation for derived facts

Table facts remain an optional recall layer and are downranked behind primary evidence.

### Evaluation

- Preserved the original 18-question benchmark.
- Added `data/petroleum/queries-50.jsonl` with 50 development questions.
- Added retrieval miss analysis and corrected several labels.
- Added answer-level evaluation in `llm/answer_quality.py` and `scripts/evaluate_answers.py`.
- Added checks for citations, sentence support, unsupported claims, numbers, units, and abstention.
- Generated 50 extractive answer drafts, but these are pipeline-validation artifacts, not independently reviewed gold answers.
- Added `scripts/run_petroleum_production_benchmark.py` to freeze and rerun the production configuration.

### Current production metrics

On the 50-question benchmark, with source-aware production retrieval, after adding source #27:

```text
Hit@5:    0.980
Recall@5: 0.981
MRR@5:    0.759
```

This is a minor, documented MRR drop from the 26-source baseline:

```text
Hit@5:    0.980
Recall@5: 0.981
MRR@5:    0.761
```

### GPU experiments

GPU retrieval is functional but not selected for production:

- Dense `all-MiniLM-L6-v2`: Hit@5 0.720, MRR 0.460.
- Best dense/BM25 fusion: Hit@5 0.860, MRR 0.611.
- Cross-encoder, candidate-k=25: Hit@5 0.860, MRR 0.638.

Conclusion: retain GPU retrieval as an experiment/candidate generator only. Do not replace the lexical pipeline without metric and latency evidence.

### Reports and documentation

Important result reports are in `reports/results/`, including:

- petroleum 50-question evaluation
- miss analysis
- GPU retrieval results
- expansion reports for sources 22 through 26
- production benchmark documentation

## Important limitations

1. The corpus now has **50 verified USGS sources**; the 50-question benchmark is separate from source count.
2. The 50-answer file exists; **43/50 are machine-verified and 7 remain flagged for human review**.
3. Corpus expansion was allowed to continue despite retrieval regression because this is a learning corpus; the best-known retrieval baseline remains recorded separately.
4. One PDF page extraction failure remains isolated for review/OCR.
5. Source #25, a short overview report, caused retrieval noise before source-aware weighting; source-aware weighting restored the baseline.
6. Dense and cross-encoder retrieval currently underperform the lexical system.
7. QLoRA must not begin yet.

## Future plan

### Phase 1 — Freeze and validate

1. Keep the 18-question and 50-question benchmarks fixed.
2. Rerun the production benchmark after every retrieval or corpus change.
3. Independently review all 50 answer records.
4. Verify citations, page numbers, numbers, units, support, and abstention labels.
5. Add adversarial unsupported and ambiguous questions.

### Phase 2 — Resolve retrieval misses

1. Investigate the remaining difficult cases, especially the Fayetteville shallow-aquifer effects question.
2. Improve table and summary-page ranking generically rather than adding query-specific rules.
3. Track per-question regressions and preserve the best known baseline.

### Phase 3 — Expand the corpus

The 50-source corpus is complete. Future expansion is optional. For any new source:

1. Use an official USGS/EIA/BSEE/OSTI URL with clear licensing.
2. Record the checksum and provenance.
3. Run the batch pipeline and preserve the benchmark result.
4. Treat retrieval regression as a tuning signal, not a corpus-build blocker.

Do not add sources with uncertain URLs, missing checksums, or unclear provenance.

### Phase 4 — Structured source routing

Add explicit routing and validation for:

- USGS: geology and resource assessments
- EIA: current energy statistics
- BSEE: offshore wells, operators, and production facts

Add date-aware retrieval, unit normalization, numeric consistency checks, and fail-closed behavior for conflicting sources.

### Phase 5 — Production packaging

Create a stable production interface with:

- one ingestion command
- one benchmark command
- one answer-evaluation command
- one documented configuration
- reproducible dependency files
- citation/provenance checks
- latency and resource measurements
- final runbook and regression report

### Phase 6 — QLoRA decision

Do not prepare QLoRA data until retrieval and answer-quality gates are stable. If fine-tuning is later justified, require:

- licensed source material
- provenance-linked examples
- independently reviewed answers
- train/validation/test separation
- abstention examples
- numeric and citation-grounding examples

QLoRA is optional; a strong retrieval and grounding pipeline is the priority.

## Recommended next action

Continue automatic source expansion with the next official, checksum-verifiable USGS/OSTI report. After each source, run:

```bash
conda run -n gpu-test python -m scripts.validate_petroleum_manifest \
  --manifest data/petroleum/manifest.jsonl \
  --raw-dir data/petroleum/raw

conda run -n gpu-test python -m unittest discover -s tests -q
```

Then rebuild the corpus and run:

```bash
conda run -n gpu-test python -m scripts.run_petroleum_production_benchmark \
  --documents <augmented-chunks.jsonl> \
  --queries data/petroleum/queries-50.jsonl \
  --output <benchmark-result.json>
```

The current acceptance rule is: preserve provenance, pass tests, and do not regress the production benchmark without a documented reason and mitigation.

## Automation update (2026-10-05)

Phase 3 is now automated end-to-end by `scripts/run_petroleum_expansion.py`:

```bash
conda run -n gpu-test python -m scripts.run_petroleum_expansion --selfcheck          # rebuild + verify baseline
conda run -n gpu-test python -m scripts.run_petroleum_expansion --auto 1 --commit    # discover, add, gate, commit, push
conda run -n gpu-test python -m scripts.run_petroleum_expansion --auto 5 --commit    # batch expansion
```

- Discovery: USGS Publications Warehouse API first (series/DOI/PDF provenance gates; numbered series only), OSTI API fallback (USGS-publisher + no-journal gates).
- Gates per source: checksum, manifest validation, 50-question benchmark regression vs `reports/results/petroleum-production-baseline.json`, full test suite, then commit+push (rebase-safe).
- Failures auto-rollback (manifest + raw PDF) and are recorded with per-query miss diagnostics in `data/petroleum/auto-expansion-rejected.json` (Phase-2 evidence).
- Corpus build complete at **50 sources, 3,249 pages, 8,490 clean chunks, 8,828 augmented chunks**.
- Final expanded-corpus retrieval: Hit@5 0.920, Recall@5 0.923, MRR@5 0.736. The original best-known baseline remains Hit@5 0.980 / Recall@5 0.981 / MRR@5 0.761; retrieval tuning is now a separate Phase-2 task.
- Local Ollama end-to-end test (`mistral:latest`, RTX 3060): **50/50 cited or abstained correctly, average 3.4 seconds/question**; report: `reports/results/petroleum-ollama-50q-final.jsonl`.
