#!/usr/bin/env python3
"""Retrieve citation-labelled evidence or explicitly abstain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.rag import (
    BM25Retriever,
    assemble_context,
    chunk_document,
    load_jsonl_chunks,
    load_jsonl_documents,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=160)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--max-characters", type=int, default=4000)
    parser.add_argument("--minimum-score", type=float, default=0.0)
    parser.add_argument(
        "--pre-chunked",
        action="store_true",
        help="load chunk records directly instead of chunking document records",
    )
    parser.add_argument(
        "--expand-petroleum-query",
        action="store_true",
        help="add conservative petroleum synonyms to the query",
    )
    parser.add_argument("--section-aware", action="store_true")
    parser.add_argument("--summary-aware", action="store_true")
    parser.add_argument("--semantic-rerank", action="store_true")
    parser.add_argument("--prefer-primary-evidence", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pre_chunked:
        chunks = load_jsonl_chunks(args.documents)
    else:
        documents = load_jsonl_documents(args.documents)
        chunks = [
            chunk
            for document in documents
            for chunk in chunk_document(
                document, chunk_size=args.chunk_size, overlap=args.overlap
            )
        ]
    result = assemble_context(
        BM25Retriever(chunks),
        args.query,
        top_k=args.top_k,
        max_characters=args.max_characters,
        minimum_score=args.minimum_score,
        expand_query=args.expand_petroleum_query,
        section_aware=args.section_aware,
        summary_aware=args.summary_aware,
        semantic_rerank=args.semantic_rerank,
        prefer_primary_evidence=args.prefer_primary_evidence,
    )
    payload = {
        "query": result.query,
        "abstained": result.abstained,
        "reason": result.reason,
        "citations": list(result.citations),
        "context": result.context,
        "results": [
            {
                "citation": result.citations[index],
                "document_id": item.chunk.document_id,
                "chunk_id": item.chunk.chunk_id,
                "source": item.chunk.source,
                "title": item.chunk.title,
                "score": item.score,
            }
            for index, item in enumerate(result.results)
        ],
    }
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if result.abstained:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
