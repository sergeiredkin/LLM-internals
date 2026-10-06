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


def source_id(record: dict) -> str:
    metadata = record.get("metadata") or {}
    value = metadata.get("source_id") or str(record.get("document_id", "")).split("#", 1)[0]
    return str(value)


def jsonl_summary(path: Path) -> tuple[int, set[str]]:
    if not path.exists():
        return 0, set()
    count = 0
    sources: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            count += 1
            source = source_id(record)
            if source:
                sources.add(source)
    return count, sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA / "manifest.jsonl")
    parser.add_argument("--raw-dir", type=Path, default=DATA / "raw")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--expected-sources", type=int, default=50)
    parser.add_argument("--minimum-table-facts", type=int, default=50)
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

    expected_ids = set(ids)
    source_artifacts = {
        "pages-expanded.jsonl": args.expected_sources,
        "chunks-expanded-clean.jsonl": args.expected_sources,
        "chunks-augmented.jsonl": args.expected_sources,
    }
    required = {
        **source_artifacts,
        "table-fact-candidates.jsonl": args.minimum_table_facts,
        "queries-50.jsonl": 50,
        "gold-answers-50.jsonl": 50,
    }
    counts = {}
    artifact_sources: dict[str, set[str]] = {}
    for name, minimum in required.items():
        value, sources = jsonl_summary(args.data_dir / name)
        counts[name] = value
        artifact_sources[name] = sources
        if value < minimum:
            errors.append(f"{name}: {value} records; expected at least {minimum}")

    for name in source_artifacts:
        missing_sources = expected_ids - artifact_sources[name]
        unknown_sources = artifact_sources[name] - expected_ids
        if missing_sources:
            errors.append(f"{name}: missing {len(missing_sources)} manifest source(s)")
        if unknown_sources:
            errors.append(f"{name}: contains {len(unknown_sources)} unknown source(s)")

    table_unknown_sources = artifact_sources["table-fact-candidates.jsonl"] - expected_ids
    if table_unknown_sources:
        errors.append(f"table-fact-candidates.jsonl: contains {len(table_unknown_sources)} unknown source(s)")
    if counts["chunks-augmented.jsonl"] < counts["chunks-expanded-clean.jsonl"] + counts["table-fact-candidates.jsonl"]:
        errors.append("chunks-augmented.jsonl does not include all clean chunks and table facts")

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
