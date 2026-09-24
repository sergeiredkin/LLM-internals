#!/usr/bin/env python3
"""Build an extractive, provenance-linked draft answer set for manual review."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def extract_answer(text: str, query: str) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    terms = {term.lower() for term in re.findall(r"[A-Za-z]{4,}", query)}
    ranked = sorted(sentences, key=lambda sentence: (-len(terms & set(re.findall(r"[A-Za-z]{4,}", sentence.lower()))), len(sentence)))
    return " ".join(ranked[:2]) if ranked else text[:500]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pages = {record["document_id"]: record for line in args.pages.open(encoding="utf-8") if line.strip() for record in [json.loads(line)]}
    rows = []
    for line in args.queries.open(encoding="utf-8"):
        if not line.strip():
            continue
        query = json.loads(line)
        ids = list(query["relevant_document_ids"])
        evidence = [pages[doc_id] for doc_id in ids if doc_id in pages]
        answer = extract_answer(" ".join(item.get("text", "") for item in evidence), query["query"])
        rows.append({
            "query": query["query"],
            "answer": answer,
            "citations": ids,
            "evidence": [{"document_id": item["document_id"], "text": item.get("text", "")} for item in evidence],
            "draft": True,
            "review_required": True,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Draft answers: {len(rows)}")
    print(f"Output:        {args.output}")


if __name__ == "__main__":
    main()
