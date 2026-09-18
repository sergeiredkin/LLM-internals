#!/usr/bin/env python3
"""Create an auditable first-pass label file for petroleum review samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.petroleum_quality import quality_flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    accepted = rejected = 0
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            flags = quality_flags(record.get("text", ""))
            record["review"] = {
                "decision": "reject" if flags else "accept",
                "quality_flags": flags,
                "reviewer": "automated-first-pass",
                "notes": "Manual confirmation required before production retrieval.",
            }
            record["review_line"] = line_number
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            if flags:
                rejected += 1
            else:
                accepted += 1
    print(f"Accepted: {accepted}")
    print(f"Rejected: {rejected}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
