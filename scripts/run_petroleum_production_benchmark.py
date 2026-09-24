#!/usr/bin/env python3
"""Run the frozen production petroleum retrieval configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.rag import BM25Retriever, load_jsonl_chunks, retrieval_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    chunks = load_jsonl_chunks(args.documents)
    queries = []
    for line in args.queries.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            queries.append((record["query"], set(record["relevant_document_ids"])))
    metrics = retrieval_metrics(
        BM25Retriever(chunks), queries, top_k=args.top_k, expand_query=True,
        prefer_primary_evidence=True, section_aware=True, summary_aware=True,
        semantic_rerank=True,
    )
    result = {"configuration": "production-lexical-v1", "chunks": len(chunks), "queries": len(queries), "top_k": args.top_k, "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
