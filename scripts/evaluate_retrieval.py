#!/usr/bin/env python3
"""Evaluate deterministic BM25 retrieval on a JSONL corpus and labelled queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.rag import (
    BM25Retriever,
    chunk_document,
    load_jsonl_chunks,
    load_jsonl_documents,
    retrieval_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=160)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--pre-chunked",
        action="store_true",
        help="load chunk records directly instead of chunking document records",
    )
    parser.add_argument(
        "--expand-petroleum-query",
        action="store_true",
        help="add conservative petroleum synonyms to each query",
    )
    parser.add_argument(
        "--prefer-primary-evidence",
        action="store_true",
        help="slightly downrank derived table facts behind source chunks",
    )
    parser.add_argument(
        "--section-aware",
        action="store_true",
        help="boost evidence passages matching the query's petroleum section intent",
    )
    parser.add_argument(
        "--summary-aware",
        action="store_true",
        help="boost query-matching abstract/summary openings",
    )
    parser.add_argument(
        "--semantic-rerank",
        action="store_true",
        help="rerank using query-term coverage in text, title, and opening context",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pre_chunked:
        chunks = load_jsonl_chunks(args.documents)
        document_count = len({chunk.document_id for chunk in chunks})
    else:
        documents = load_jsonl_documents(args.documents)
        document_count = len(documents)
        chunks = [
            chunk
            for document in documents
            for chunk in chunk_document(
                document, chunk_size=args.chunk_size, overlap=args.overlap
            )
        ]
    with args.queries.open(encoding="utf-8") as handle:
        queries = []
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
                queries.append((str(record["query"]), set(map(str, record["relevant_document_ids"]))))
            except (KeyError, TypeError, ValueError) as error:
                raise SystemExit(f"invalid query JSONL at line {line_number}") from error

    retriever = BM25Retriever(chunks)
    metrics = retrieval_metrics(
        retriever,
        queries,
        top_k=args.top_k,
        expand_query=args.expand_petroleum_query,
        prefer_primary_evidence=args.prefer_primary_evidence,
        section_aware=args.section_aware,
        summary_aware=args.summary_aware,
        semantic_rerank=args.semantic_rerank,
    )
    result = {
        "documents": document_count,
        "chunks": len(chunks),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "top_k": args.top_k,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Documents:       {document_count}")
    print(f"Chunks:           {len(chunks)}")
    print(f"Hit rate@{args.top_k}:   {metrics['hit_rate_at_k']:.3f}")
    print(f"Recall@{args.top_k}:     {metrics['recall_at_k']:.3f}")
    print(f"MRR@{args.top_k}:        {metrics['mrr_at_k']:.3f}")
    print(f"Result:           {args.output}")


if __name__ == "__main__":
    main()
