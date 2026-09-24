#!/usr/bin/env python3
"""Extract table-like petroleum pages into a separate provenance-preserving corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.petroleum_quality import table_candidate_score
from llm.rag import Document, chunk_document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-score", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=160)
    parser.add_argument("--overlap", type=int, default=32)
    args = parser.parse_args()
    if args.minimum_score <= 0:
        raise SystemExit("--minimum-score must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = 0
    with args.pages.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as output:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            score = table_candidate_score(str(record.get("text", "")))
            if score < args.minimum_score:
                continue
            document = Document(
                document_id=str(record["document_id"]),
                text=str(record.get("text", "")),
                source=str(record.get("source", "")),
                title=str(record.get("title", "")),
                metadata={
                    **{str(k): str(v) for k, v in (record.get("metadata") or {}).items()},
                    "record_type": "table_candidate",
                    "table_candidate_score": str(score),
                    "input_line": str(line_number),
                },
            )
            table_chunks = chunk_document(
                document, chunk_size=args.chunk_size, overlap=args.overlap
            )
            for table_chunk in table_chunks:
                output.write(json.dumps({
                    "chunk_id": table_chunk.chunk_id.replace("#chunk-", "#table-chunk-"),
                    "document_id": table_chunk.document_id,
                    "text": table_chunk.text,
                    "source": table_chunk.source,
                    "title": table_chunk.title,
                    "metadata": table_chunk.metadata,
                }, ensure_ascii=False) + "\n")
            selected += 1
    print(f"Table candidate pages: {selected}")
    print(f"Output:               {args.output}")


if __name__ == "__main__":
    main()
