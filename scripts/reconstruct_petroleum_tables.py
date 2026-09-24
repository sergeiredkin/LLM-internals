#!/usr/bin/env python3
"""Create auditable row-like records from merged petroleum table pages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.petroleum_quality import reconstruct_table_rows, table_candidate_score
from scripts.extract_petroleum_tables import merge_table_continuations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-score", type=int, default=10)
    args = parser.parse_args()
    if args.minimum_score <= 0:
        raise SystemExit("--minimum-score must be positive")

    records = [json.loads(line) for line in args.pages.open(encoding="utf-8") if line.strip()]
    records = merge_table_continuations(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as output:
        for record in records:
            text = str(record.get("text", ""))
            if table_candidate_score(text) < args.minimum_score:
                continue
            metadata = {str(k): str(v) for k, v in (record.get("metadata") or {}).items()}
            metadata.update({
                "record_type": "table_row_candidate",
                "table_page_range": ",".join(map(str, record.get("table_page_range", [metadata.get("page", "")]))),
            })
            for row_number, row in enumerate(reconstruct_table_rows(text)):
                output.write(json.dumps({
                    "chunk_id": f"{record['document_id']}#table-row-{row_number:04d}",
                    "document_id": record["document_id"],
                    "text": row,
                    "source": record.get("source", ""),
                    "title": record.get("title", ""),
                    "metadata": metadata,
                }, ensure_ascii=False) + "\n")
                count += 1
    print(f"Table row candidates: {count}")
    print(f"Output:              {args.output}")


if __name__ == "__main__":
    main()
