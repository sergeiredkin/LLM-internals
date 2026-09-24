#!/usr/bin/env python3
"""Extract table-like petroleum pages into a separate provenance-preserving corpus."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from llm.petroleum_quality import table_candidate_score
from llm.rag import Document, chunk_document


def merge_table_continuations(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Merge a table page with nearby pages labelled as its continuation."""

    merged: list[dict[str, object]] = []
    index = 0
    while index < len(records):
        record = records[index]
        text = str(record.get("text", ""))
        if not re.search(r"\btable\s+\d+\s*[.:]", text, re.IGNORECASE):
            merged.append(record)
            index += 1
            continue
        source_id = str((record.get("metadata") or {}).get("source_id", ""))
        end = index
        for lookahead in range(index + 1, min(index + 4, len(records))):
            candidate = records[lookahead]
            candidate_source = str((candidate.get("metadata") or {}).get("source_id", ""))
            candidate_text = str(candidate.get("text", ""))
            if candidate_source != source_id:
                break
            if re.search(r"\btable\s+\d+[\s.:,-]*(?:con['’]?t|continued)", candidate_text, re.IGNORECASE):
                end = lookahead
                break
        if end == index:
            merged.append(record)
            index += 1
            continue
        combined = dict(record)
        combined["text"] = "\n".join(str(records[pos].get("text", "")) for pos in range(index, end + 1))
        combined["table_page_range"] = [
            str((records[pos].get("metadata") or {}).get("page", ""))
            for pos in range(index, end + 1)
        ]
        merged.append(combined)
        index = end + 1
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-score", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=160)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument(
        "--merge-continuations",
        action="store_true",
        help="merge nearby pages containing a labelled table continuation",
    )
    args = parser.parse_args()
    if args.minimum_score <= 0:
        raise SystemExit("--minimum-score must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = 0
    records = [json.loads(line) for line in args.pages.open(encoding="utf-8") if line.strip()]
    if args.merge_continuations:
        records = merge_table_continuations(records)
    with args.output.open("w", encoding="utf-8") as output:
        for line_number, record in enumerate(records, 1):
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
                    **({"table_page_range": ",".join(map(str, record["table_page_range"]))}
                       if record.get("table_page_range") else {}),
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
