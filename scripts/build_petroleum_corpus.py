#!/usr/bin/env python3
"""Build the reproducible petroleum RAG corpus from the manifest.

This is the stable release command. It deliberately does not apply a retrieval
quality gate: corpus construction and retrieval evaluation are separate steps.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "petroleum"


def run(module: str, *args: object) -> None:
    command = [sys.executable, "-m", f"scripts.{module}", *(str(arg) for arg in args)]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA / "manifest.jsonl")
    parser.add_argument("--raw-dir", type=Path, default=DATA / "raw")
    parser.add_argument("--output-dir", type=Path, default=DATA)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    pages = out / "pages-expanded.jsonl"
    chunks = out / "chunks-expanded.jsonl"
    review = out / "review-build.jsonl"
    clean = out / "chunks-expanded-clean.jsonl"
    rejected = out / "chunks-expanded-rejected.jsonl"
    table_candidates = out / "table-candidates.jsonl"
    table_rows = out / "table-row-candidates.jsonl"
    table_facts = out / "table-fact-candidates.jsonl"
    augmented = out / "chunks-augmented.jsonl"

    # Ingestion downloads missing PDFs and verifies every manifest checksum.
    run("ingest_petroleum_pdfs", "--manifest", args.manifest, "--raw-dir", args.raw_dir, "--output", pages)
    run("validate_petroleum_manifest", "--manifest", args.manifest, "--raw-dir", args.raw_dir)
    run("prepare_petroleum_chunks", "--pages", pages, "--chunks", chunks, "--review", review)
    run("filter_petroleum_chunks", "--input", chunks, "--clean", clean, "--rejected", rejected)
    run("extract_petroleum_tables", "--pages", pages, "--output", table_candidates, "--merge-continuations")
    run("reconstruct_petroleum_tables", "--pages", pages, "--output", table_rows)
    run("construct_petroleum_table_facts", "--rows", table_rows, "--output", table_facts)
    run("build_petroleum_augmented", "--base", clean, "--table-facts", table_facts, "--output", augmented)

    summary = {
        "sources": sum(1 for line in args.manifest.open(encoding="utf-8") if line.strip()),
        "pages": count(pages),
        "chunks": count(chunks),
        "clean_chunks": count(clean),
        "rejected_chunks": count(rejected),
        "table_facts": count(table_facts),
        "augmented_chunks": count(augmented),
        "output": str(augmented),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
