# Petroleum RAG Week 1 Pilot

**Date:** 2026-09-18

## Corpus

The first pilot uses 12 USGS petroleum reports selected from public USGS publication pages. The
manifest records the source URL, title, publisher, publication year, license status, and downloaded
file checksum.

- Sources: **12 reports**
- Extracted pages: **1,460**
- Text chunks: **3,303**
- Review sample: **50 chunks**, evenly distributed across the corpus
- Extraction failures: **1 page**

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

The 50-sample review found approximately 12 suspect samples by simple inspection:

- figure/map-only pages with little useful text
- legacy table text with corrupted symbols or column order
- references/bibliography passages
- title-only or page-header-only chunks
- one malformed PDF page that pypdf could not parse

The ingestion script continued safely, preserved the PDF checksum, and reported the failed page:

```text
usgs-of-1994-0559-south-america page 189
```

That page must be OCR-processed or manually recovered before production use.

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
```

## Decision

The pipeline is suitable for expanding the report collection, but the current chunks are not yet
production-quality evidence. First improve quality filtering and recover the failed page. Then add
EIA structured data and BSEE records without mixing them into the report-text index.
