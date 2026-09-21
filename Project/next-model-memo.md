# Handoff Memo: Petroleum RAG Next Model

**Repository:** `/home/sergei/Documents/learngpt`
**Branch:** `main`
**Remote:** `origin/main`
**Latest commit:** `0f901af Validate expanded petroleum corpus`
**Working tree:** clean
**Environment:** `gpu-test`
**Hardware:** RTX 3060 12 GB

## User instruction

Do **not** expand beyond the current 20-report corpus today. The user explicitly wants expansion toward
50 reports tomorrow, not now.

## Current corpus

Manifest:

```text
data/petroleum/manifest.jsonl
```

The manifest contains **20 checksum-verified USGS reports**. Eight were added in the latest expansion:

- USGS SIR 2012-5008: coalbed natural gas produced water
- USGS SIR 2012-5041: Piceance Basin oil shale
- USGS SIR 2012-5046: Gulf of Mexico gas hydrates
- USGS SIR 2012-5054: Indian National Gas Hydrate Program
- USGS SIR 2012-5076: Green River oil shale
- USGS SIR 2012-5145: Cook Inlet-Susitna coal-bed gas
- USGS SIR 2012-5159: Austin Chalk oil and gas
- USGS SIR 2012-5273: Fayetteville Shale gas area

Local ignored data:

```text
20 reports
1,838 recovered pages
4,613 original chunks
4,069 clean chunks
544 rejected chunks
100 review samples
```

Raw PDFs are in `data/petroleum/raw/` and currently occupy approximately 221 MB.

The malformed South America PDF page 189 was visually inspected. It is a rotated stratigraphic chart,
not prose. It was recovered as a caption-only record using:

```text
extraction: manual-figure-caption
page_status: figure_only
```

Recovery utility:

```bash
conda run -n gpu-test python -m scripts.recover_petroleum_page \
  --manifest data/petroleum/manifest.jsonl \
  --pages data/petroleum/pages-expanded.jsonl \
  --chunks data/petroleum/chunks-expanded.jsonl \
  --source-id usgs-of-1994-0559-south-america \
  --page 189 \
  --printed-page 171 \
  --caption "Figure 95. Stratigraphic chart of the Amazonas basin (from Neves, 1990)."
```

## Quality review

The original 50 samples were manually confirmed earlier. The expanded 100-sample review was also
confirmed:

```text
84 prose candidates
16 rejected
```

The tracked overrides are:

```text
data/petroleum/review-expanded-overrides.json
```

Generated labeled review data is ignored:

```text
data/petroleum/review-expanded-100-labeled.jsonl
```

The 16 rejected samples are figure/map/chart pages, bibliography/title material, or unusable symbol
extraction. Do not mix them into ordinary prose retrieval.

## Retrieval status

Fixed questions:

```text
data/petroleum/queries.jsonl
```

This now contains **18 questions**: the original 10 plus one question for each of the 8 added reports.

Expanded clean corpus:

```text
data/petroleum/chunks-expanded-clean.jsonl
```

BM25 evaluation command:

```bash
cd ~/Documents/learngpt

conda run -n gpu-test python -m scripts.evaluate_retrieval \
  --documents data/petroleum/chunks-expanded-clean.jsonl \
  --queries data/petroleum/queries.jsonl \
  --output reports/results/petroleum-bm25-expanded.json \
  --chunk-size 160 \
  --overlap 32 \
  --top-k 5
```

Current expanded metrics:

```text
Hit rate@5: 0.722
Recall@5:   0.737
MRR@5:      0.479
```

Miss inspection:

```bash
conda run -n gpu-test python -m scripts.inspect_retrieval_misses \
  --documents data/petroleum/chunks-expanded-clean.jsonl \
  --queries data/petroleum/queries.jsonl \
  --output reports/results/petroleum-bm25-expanded-misses.json \
  --top-k 5
```

There are **5 misses** in the expanded 18-question evaluation. Inspect these before adding reports.
The lower score is useful and should not be hidden by changing questions casually.

Retrieval context now preserves:

- source URL
- source ID
- PDF page number
- chunk ID
- abstention when no positive lexical evidence exists

Citation implementation is in `llm/rag.py`; all tests cover page/source citation output.

## EIA status

Local normalized database:

```text
data/petroleum/eia.sqlite
```

SQLite is used because PostgreSQL is not installed. The schema is relational and portable.

Current pilot data contains approximately 2,000 observations:

- crude production
- inventories
- refinery inputs
- spot prices

The EIA key is stored locally in ignored `.env`:

```text
EIA_API_KEY=...
```

Never print or commit the key. `.env.example` is tracked.

EIA commands:

```bash
conda run -n gpu-test python -m scripts.ingest_eia \
  --database data/petroleum/eia.sqlite \
  --start 2024-01-01 \
  --end 2025-12-31 \
  --length 500 \
  --continue-on-error
```

```bash
conda run -n gpu-test python -m scripts.query_eia \
  --database data/petroleum/eia.sqlite \
  --series MCRFPUS1 \
  --yoy-period 2025-01
```

## BSEE status

No BSEE records are loaded yet. A strict importer exists:

```text
llm/bsee.py
scripts/ingest_bsee.py
```

It requires explicitly mapped CSV columns with oil/gas/water units, source URL, and checksum. Do not
automatically interpret BSEE ranked fixed-width aggregate files without verifying their column meanings.

## Tests

Current test count:

```text
116 tests passing
```

Run:

```bash
cd ~/Documents/learngpt
conda run -n gpu-test python -m unittest discover -s tests -q
```

## Recommended next action today

The user said not to expand to 50 reports yet. Therefore:

1. Read `reports/results/petroleum-bm25-expanded-misses.json`.
2. Classify each miss as query wording, chunk boundary, extraction problem, or missing evidence.
3. Do not silently alter the fixed questions to improve metrics.
4. If appropriate, add query variants as a separate evaluation file rather than replacing the fixed set.
5. Verify citation and abstention behavior on the 18-question set.
6. Commit only evaluation/retrieval improvements; do not add more source reports today.

## Tomorrow

After the five misses are understood, expand from 20 toward 50 verified public/open USGS reports. Keep
manifest-first ingestion, checksums, page provenance, quality filtering, and review sampling. Do not
start QLoRA until retrieval, citations, units, and abstention are stable.

## Important constraints

- Keep raw PDFs, generated JSONL, databases, and `.env` out of Git.
- Preserve source IDs, URLs, retrieval dates, checksums, pages, units, and licenses.
- Do not claim educational dequantization kernels are faster.
- Do not use questionable OnePetro material.
- Separate explanatory report retrieval from structured EIA/BSEE numerical truth.
- Use isolated code blocks for runnable commands.
