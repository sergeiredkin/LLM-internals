#!/usr/bin/env python3
"""Chunk extracted petroleum pages and create a deterministic manual-review sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.rag import Chunk, Document, chunk_document, load_jsonl_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=160)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--review-count", type=int, default=50)
    return parser.parse_args()


def sample_chunks(chunks: list[Chunk], count: int) -> list[Chunk]:
    if count <= 0:
        raise ValueError("review count must be positive")
    if len(chunks) <= count:
        return chunks
    # Evenly spread review across the corpus rather than selecting only the first report.
    indices = [round(index * (len(chunks) - 1) / (count - 1)) for index in range(count)]
    return [chunks[index] for index in indices]


def main() -> None:
    args = parse_args()
    if args.review_count <= 0:
        raise SystemExit("--review-count must be positive")
    documents = load_jsonl_documents(args.pages)
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document, chunk_size=args.chunk_size, overlap=args.overlap)
    ]
    args.chunks.parent.mkdir(parents=True, exist_ok=True)
    args.review.parent.mkdir(parents=True, exist_ok=True)
    with args.chunks.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "source": chunk.source,
                "title": chunk.title,
                "metadata": chunk.metadata,
            }, ensure_ascii=False) + "\n")
    with args.review.open("w", encoding="utf-8") as output:
        for chunk in sample_chunks(chunks, args.review_count):
            output.write(json.dumps({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "source": chunk.source,
                "title": chunk.title,
                "metadata": chunk.metadata,
                "review": {
                    "text_complete": None,
                    "headers_clean": None,
                    "table_quality": None,
                    "ocr_quality": None,
                    "duplicate_boilerplate": None,
                    "notes": "",
                },
            }, ensure_ascii=False) + "\n")
    print(f"Pages:          {len(documents)}")
    print(f"Chunks:         {len(chunks)}")
    print(f"Review samples: {min(len(chunks), args.review_count)}")
    print(f"Chunks output:  {args.chunks}")
    print(f"Review output:  {args.review}")


if __name__ == "__main__":
    main()
