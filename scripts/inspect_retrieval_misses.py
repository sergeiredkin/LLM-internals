#!/usr/bin/env python3
"""Report fixed-query BM25 misses with their top retrieved passages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.rag import BM25Retriever, Chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.documents.open(encoding="utf-8") if line.strip()]
    chunks = [Chunk(chunk_id=row["chunk_id"], document_id=row["document_id"], text=row["text"],
                    source=row["source"], title=row["title"], metadata=row["metadata"])
              for row in rows]
    retriever = BM25Retriever(chunks)
    misses = []
    for query in (json.loads(line) for line in args.queries.open(encoding="utf-8") if line.strip()):
        results = retriever.retrieve(query["query"], top_k=args.top_k)
        relevant = set(query["relevant_document_ids"])
        if any(result.chunk.document_id in relevant for result in results):
            continue
        misses.append({
            "query": query["query"],
            "relevant_document_ids": sorted(relevant),
            "results": [{"document_id": result.chunk.document_id, "score": result.score,
                          "text": result.chunk.text, "source": result.chunk.source,
                          "page": (result.chunk.metadata or {}).get("page")}
                         for result in results],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(misses, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Misses: {len(misses)}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
