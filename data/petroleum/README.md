# Petroleum-domain corpus

This directory is reserved for a documented petroleum-domain corpus. No source documents are
committed yet.

## Required manifest

Before adding data, record for every source:

- title and publisher
- URL or archive identifier
- retrieval date
- license and redistribution terms
- checksum
- document type and language
- any processing or OCR steps

Recommended normalized JSONL schema:

```json
{"document_id":"api-example-001","text":"...","source":"...","title":"...","metadata":{"license":"CC-BY-4.0","year":"2024","topic":"drilling"}}
```

Keep raw downloads and processed corpora out of Git. The pilot pipeline is:

```bash
python -m scripts.ingest_petroleum_pdfs \
  --manifest data/petroleum/manifest.jsonl \
  --output data/petroleum/pages.jsonl

python -m scripts.prepare_petroleum_chunks \
  --pages data/petroleum/pages.jsonl \
  --chunks data/petroleum/chunks.jsonl \
  --review data/petroleum/review-50.jsonl
```

The retrieval code accepts JSONL through `scripts/evaluate_retrieval.py` and preserves document/source
metadata through chunking. Review `review-50.jsonl` before scaling beyond the pilot.

Do not use proprietary manuals, standards, or paid reports unless redistribution and model-training
rights are explicitly confirmed.
