#!/usr/bin/env python3
"""Download checksum-tracked petroleum PDFs and extract page-aware JSONL documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/petroleum/raw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-agent", default="learnGPT-petroleum-research/1.0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(source_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id).strip("._") or "source"


def download(url: str, destination: Path, user_agent: str) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise SystemExit(f"manifest does not exist: {args.manifest}")
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise SystemExit("Install pypdf before PDF ingestion: python -m pip install pypdf") from error

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_count = 0
    source_count = 0
    failed_pages = 0
    with args.manifest.open(encoding="utf-8") as manifest, args.output.open("w", encoding="utf-8") as output:
        for line_number, line in enumerate(manifest, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                source_id = str(record["source_id"])
                url = str(record["url"])
                title = str(record.get("title", source_id))
                license_name = str(record["license"])
            except (KeyError, TypeError, ValueError) as error:
                raise SystemExit(f"invalid source manifest line {line_number}") from error
            if not url.lower().endswith(".pdf") and record.get("format", "pdf") != "pdf":
                raise SystemExit(f"source {source_id} is not marked as a PDF")
            destination = args.raw_dir / f"{safe_name(source_id)}.pdf"
            expected_hash = record.get("sha256")
            if destination.exists() and expected_hash and sha256(destination) != expected_hash:
                raise SystemExit(f"existing file checksum mismatch: {destination}")
            if not destination.exists():
                print(f"Downloading {source_id}: {url}")
                temporary = destination.with_suffix(".pdf.tmp")
                download(url, temporary, args.user_agent)
                temporary.replace(destination)
            actual_hash = sha256(destination)
            if expected_hash and actual_hash.lower() != str(expected_hash).lower():
                raise SystemExit(f"checksum mismatch for {source_id}: {actual_hash}")

            reader = PdfReader(str(destination), strict=False)
            for page_number, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text() or ""
                except Exception as error:  # malformed legacy PDFs need a later OCR pass
                    failed_pages += 1
                    print(
                        f"WARN {source_id} page {page_number}: extraction failed: "
                        f"{type(error).__name__}: {error}"
                    )
                    continue
                text = re.sub(r"[ \t]+", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text).strip()
                if not text:
                    continue
                document = {
                    "document_id": f"{source_id}#page-{page_number:04d}",
                    "text": text,
                    "source": url,
                    "title": title,
                    "metadata": {
                        "source_id": source_id,
                        "license": license_name,
                        "publication_date": str(record.get("publication_date", "")),
                        "publisher": str(record.get("publisher", "")),
                        "page": str(page_number),
                        "sha256": actual_hash,
                        "extraction": "pypdf-text",
                        "source_role": str(record.get("source_role", "domain-report")),
                    },
                }
                output.write(json.dumps(document, ensure_ascii=False) + "\n")
                output_count += 1
            source_count += 1
            print(f"Extracted {source_id}: {len(reader.pages)} pages")
    print(f"Sources:  {source_count}")
    print(f"Pages:    {output_count}")
    print(f"Failed pages: {failed_pages}")
    print(f"Output:   {args.output}")
    if failed_pages:
        print("WARNING: failed pages were skipped; review and OCR them before production use.")


if __name__ == "__main__":
    main()
