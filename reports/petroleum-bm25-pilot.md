# Petroleum BM25 Retrieval Pilot

**Date:** 2026-09-18

## Fixed evaluation set

The tracked query set is:

`data/petroleum/queries.jsonl`

It contains 10 questions with labelled relevant USGS page-document IDs. The questions cover basin
geology, traps, reservoirs, source rocks, petroleum systems, and resource-assessment context.

## Configuration

- Corpus: `data/petroleum/chunks-clean.jsonl`
- Documents/chunks evaluated: 2,859
- Retrieval: deterministic Okapi BM25
- Top-k: 5
- Chunk size: 160 words
- Overlap: 32 words
- Fixed query count: 10

## Results

```text
Hit rate@5: 0.800
Recall@5:   0.800
MRR@5:      0.603
```

Result JSON:

`reports/results/petroleum-bm25-pilot.json`

These are retrieval-only metrics. They do not demonstrate answer correctness. The next evaluation
must inspect the retrieved passage, citation page, unit handling, and abstention behavior.

## Interpretation

The lexical baseline is useful enough to establish a comparison point, but 20 percent of the fixed
questions were not retrieved at the labelled page within the top five. Do not add dense retrieval or
QLoRA yet; first inspect the misses and determine whether they are query wording, chunk boundaries,
table extraction, or missing evidence.
