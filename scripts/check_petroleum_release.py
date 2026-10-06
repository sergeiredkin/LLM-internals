#!/usr/bin/env python3
"""Verify a built petroleum RAG release without requiring an LLM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "petroleum"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lines(path: Path) -> int:
    return sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.exists() else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA / "manifest.jsonl")
    parser.add_argument("--raw-dir", type=Path, default=DATA / "raw")
    parser.add_argument("--expected-sources", type=int, default=50)
    parser.add_argument("--require-raw", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    records = [json.loads(line) for line in args.manifest.open(encoding="utf-8") if line.strip()]
    ids = [str(record.get("source_id", "")) for record in records]
    if len(records) != args.expected_sources:
        errors.append(f"manifest has {len(records)} sources; expected {args.expected_sources}")
    if len(set(ids)) != len(ids):
        errors.append("manifest contains duplicate source_id values")

    missing = 0
    mismatched = 0
    for record in records:
        path = args.raw_dir / f"{record['source_id']}.pdf"
        if not path.exists():
            missing += 1
            continue
        if record.get("sha256") and sha256(path).lower() != str(record["sha256"]).lower():
            mismatched += 1
    if args.require_raw and missing:
        errors.append(f"{missing} raw PDFs are missing")
    if mismatched:
        errors.append(f"{mismatched} raw PDF checksums do not match")

    required = {
        "pages-expanded.jsonl": 1,
        "chunks-expanded-clean.jsonl": 1,
        "chunks-augmented.jsonl": 1,
        "queries-50.jsonl": 50,
        "gold-answers-50.jsonl": 50,
    }
    counts = {}
    for name, minimum in required.items():
        value = lines(DATA / name)
        counts[name] = value
        if value < minimum:
            errors.append(f"{name}: {value} records; expected at least {minimum}")

    result = {
        "status": "pass" if not errors else "fail",
        "sources": len(records),
        "missing_raw_pdfs": missing,
        "checksum_mismatches": mismatched,
        "artifacts": counts,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
