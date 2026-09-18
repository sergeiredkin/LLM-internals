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

Keep raw downloads and processed corpora out of Git. The retrieval code accepts JSONL through
`scripts/evaluate_retrieval.py` and preserves document/source metadata through chunking.

Do not use proprietary manuals, standards, or paid reports unless redistribution and model-training
rights are explicitly confirmed.
