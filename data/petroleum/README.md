# Petroleum-domain corpus

This directory contains the tracked release inputs for the educational petroleum RAG corpus:

- `manifest.jsonl`: 50 public USGS source records with official URLs,
  licenses/provenance, and SHA-256 checksums.
- `queries-50.jsonl`: the frozen 50-question retrieval benchmark.
- `gold-answers-50.jsonl`: provenance-linked answer records for local Ollama validation.

Raw PDFs and generated page/chunk/table JSONL artifacts stay out of Git. Build and verification
commands are documented in the [Petroleum RAG quickstart](../../Project/petroleum-rag-quickstart.md).

Do not use proprietary manuals, standards, or paid reports unless redistribution and model-training
rights are explicitly confirmed.
