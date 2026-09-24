#!/usr/bin/env python3
"""Validate petroleum manifest uniqueness, checksums, and local raw coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()
    seen: set[str] = set()
    checked = 0
    for line_number, line in enumerate(args.manifest.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        source_id = str(record["source_id"])
        if source_id in seen:
            raise SystemExit(f"duplicate source_id on line {line_number}: {source_id}")
        seen.add(source_id)
        expected = str(record.get("sha256", ""))
        if len(expected) != 64:
            raise SystemExit(f"missing or invalid sha256 for {source_id}")
        path = args.raw_dir / f"{source_id}.pdf"
        if not path.exists():
            raise SystemExit(f"missing local PDF: {path}")
        actual = digest(path)
        if actual.lower() != expected.lower():
            raise SystemExit(f"checksum mismatch for {source_id}: {actual}")
        checked += 1
    print(f"Validated sources: {checked}")
    print(f"Raw directory:    {args.raw_dir}")


if __name__ == "__main__":
    main()
