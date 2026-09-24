#!/usr/bin/env python3
"""Construct conservative, citation-preserving facts from table row candidates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def facts_from_row(record: dict) -> list[dict]:
    text = re.sub(r"\s+", " ", str(record.get("text", ""))).strip()
    lowered = text.lower()
    hydrocarbon = None
    for candidate in ("dry gas", "oil and gas", "gas", "oil"):
        if candidate in lowered:
            hydrocarbon = candidate
            break
    if not hydrocarbon:
        return []

    extent_match = re.search(
        r"(?:over a large area|in a narrow [^,.]+ band|large area)\s+([^.;]+(?:south|north|coast|barrow)[^.;]*)",
        text,
        re.IGNORECASE,
    )
    extent = extent_match.group(0).strip() if extent_match else None
    fields = [f"Hydrocarbon type: {hydrocarbon}."]
    if extent:
        fields.append(f"Areal extent: {extent}.")
    fields.append(f"Evidence: {text}")
    return [{
        "chunk_id": f"{record['chunk_id']}#fact-0000",
        "document_id": record["document_id"],
        "text": " ".join(fields),
        "source": record.get("source", ""),
        "title": record.get("title", ""),
        "metadata": {
            **(record.get("metadata") or {}),
            "record_type": "table_fact_candidate",
            "extraction_method": "conservative-regex",
            "confidence": "medium",
            "hydrocarbon_type": hydrocarbon,
        },
    }]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as output:
        for line in args.rows.open(encoding="utf-8"):
            if not line.strip():
                continue
            for fact in facts_from_row(json.loads(line)):
                output.write(json.dumps(fact, ensure_ascii=False) + "\n")
                count += 1
    print(f"Table fact candidates: {count}")
    print(f"Output:               {args.output}")


if __name__ == "__main__":
    main()
