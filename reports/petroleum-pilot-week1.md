# Petroleum RAG Week 1 Pilot

**Date:** 2026-09-18

## Corpus

The first pilot uses 12 USGS petroleum reports selected from public USGS publication pages. The
manifest records the source URL, title, publisher, publication year, license status, and downloaded
file checksum.

- Sources: **12 reports**
- Extracted/recovered pages: **1,461**
- Text chunks: **3,304**
- Review sample: **50 chunks**, evenly distributed across the corpus
- Extraction failures: **1 page**, recovered as a verified figure-only page record

The PDFs remain in ignored local storage. The tracked manifest is:

`data/petroleum/manifest.jsonl`

## Sources selected

- Amerasia Basin petroleum assessment
- Arctic Alaska petroleum assessment
- San Juan Basin petroleum geology and plays
- Petroleum exploration in Malawi
- East Siberia petroleum geology
- Undiscovered petroleum resources of Indonesia
- World petroleum resources perspective
- Naval Petroleum Reserve No. 4
- Southern South America petroleum assessment
- South Asia petroleum assessment
- Permian Basin petroleum geology and plays
- Probability and statistics for petroleum resource assessment

The collection provides basin geology, petroleum systems, plays, resource assessment, exploration
context, and assessment methodology. It is not yet an operations or market-data corpus; EIA and
BSEE remain separate structured layers.

## Extraction findings

The 50-sample review was manually confirmed by the coding agent: 30 samples are accepted for
prose retrieval and 20 are rejected or reserved for future table/figure processing. The label file
is `data/petroleum/review-50-labeled.jsonl`; it is ignored by Git because it is generated corpus
data. Decisions are reproducible through the tracked `data/petroleum/review-overrides.json` file.

- figure/map-only pages with little useful text
- legacy table text with corrupted symbols or column order
- references/bibliography passages
- title-only or page-header-only chunks
- one malformed PDF page that pypdf could not parse; it was recovered as a figure-only record

The accepted records are suitable for the BM25 pilot, but numerical claims still require citation
and unit checks. Rejected records retain provenance and can later be handled by table extraction or
OCR.

The ingestion script continued safely, preserved the PDF checksum, and reported the failed page:

```text
usgs-of-1994-0559-south-america page 189
```

The page was rendered and visually identified as a rotated stratigraphic chart. It was recovered as
a caption-only record with `extraction: manual-figure-caption` and `page_status: figure_only`; it is
excluded from prose retrieval rather than represented as fabricated text.

## Automatic quality filter

The first conservative filter processed all 3,304 chunks:

- Clean chunks: **2,859**
- Rejected chunks: **445**
- Figure/map-only flags: 367
- Too-short flags: 208
- Corruption flags: 6
- Bibliography flags: 2

Flags can overlap. Rejected chunks are preserved in `data/petroleum/chunks-rejected.jsonl`; clean
chunks are in `data/petroleum/chunks-clean.jsonl`. The filter does not reject tables merely because
they contain symbols. It is a first-pass filter, not a scientific quality guarantee.

## Policy changes before scaling

1. Keep page provenance for every chunk.
2. Exclude or separately index figure-only chunks.
3. Detect and separate references from explanatory content.
4. Add table-aware extraction rather than treating table text as prose.
5. Keep failed-page records for OCR retry; never silently treat missing pages as complete.
6. Manually label the 50 review records before expanding beyond this pilot.
7. Use separate retrieval collections for explanatory reports and structured EIA/BSEE facts.

## Reproduction

```bash
python -m scripts.ingest_petroleum_pdfs \
  --manifest data/petroleum/manifest.jsonl \
  --output data/petroleum/pages.jsonl

python -m scripts.prepare_petroleum_chunks \
  --pages data/petroleum/pages.jsonl \
  --chunks data/petroleum/chunks.jsonl \
  --review data/petroleum/review-50.jsonl

python -m scripts.filter_petroleum_chunks \
  --input data/petroleum/chunks.jsonl \
  --clean data/petroleum/chunks-clean.jsonl \
  --rejected data/petroleum/chunks-rejected.jsonl
```

## Decision

The pipeline is suitable for expanding the report collection, but the current chunks are not yet
production-quality evidence. First improve quality filtering and recover the failed page. Then add
EIA structured data and BSEE records without mixing them into the report-text index.
