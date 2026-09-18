#!/usr/bin/env python3
"""Append a manually verified figure-only recovery record for a failed PDF page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--printed-page", type=int)
    parser.add_argument("--caption", required=True)
    args = parser.parse_args()
    manifest = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    source = next((item for item in manifest if item.get("source_id") == args.source_id), None)
    if source is None:
        raise SystemExit(f"source not found: {args.source_id}")
    document_id = f"{args.source_id}#page-{args.page:04d}"
    chunk_id = f"{document_id}#chunk-0000"
    metadata = {
        "source_id": args.source_id,
        "license": source["license"],
        "publication_date": str(source.get("publication_date", "")),
        "publisher": str(source.get("publisher", "")),
        "page": str(args.page),
        "sha256": source["sha256"],
        "extraction": "manual-figure-caption",
        "page_status": "figure_only",
    }
    if args.printed_page is not None:
        metadata["printed_page"] = str(args.printed_page)
    record = {"document_id": document_id, "text": args.caption, "source": source["url"],
              "title": source.get("title", args.source_id), "metadata": metadata}
    chunk = {"chunk_id": chunk_id, **record}
    for path, item, key in [(args.pages, record, "document_id"), (args.chunks, chunk, "chunk_id")]:
        existing = {json.loads(line).get(key) for line in path.open(encoding="utf-8") if line.strip()}
        if item[key] not in existing:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Appended {item[key]} to {path}")
        else:
            print(f"Already present: {item[key]}")


if __name__ == "__main__":
    main()
