"""TinyStories download, document extraction, BPE training, and tokenization."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq

from .tokenizer import BPETokenizer


TRAIN_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/"
    "train-00000-of-00004-2d5a1467fff1081b.parquet"
)
VALIDATION_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/"
    "validation-00000-of-00001-869c898b519ad725.parquet"
)
TRAIN_SHA256 = "77cf780cebe52b6e83e3a2ac84bc56d8059363113e41d17a023f1d8b2ed0fc0b"
VALIDATION_SHA256 = "33406a6206554cfc279c29c11f4df51528af487aa1a602b075566fc83c49dcab"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: str | Path, expected_sha256: str) -> Path:
    """Download once, verify SHA-256, and atomically install the file."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and sha256_file(path) == expected_sha256:
        print(f"Using verified download: {path}")
        return path

    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "learnGPT/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        next_report = 32 * 1024 * 1024
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                suffix = f"/{total / 2**20:.0f} MiB" if total else " MiB"
                print(f"  downloaded {downloaded / 2**20:.0f}{suffix}")
                next_report += 32 * 1024 * 1024

    actual = sha256_file(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {url}: expected {expected_sha256}, got {actual}"
        )
    os.replace(temporary, path)
    return path


def extract_documents(parquet_path: str | Path, jsonl_path: str | Path, limit: int) -> int:
    """Extract non-empty text rows to document-preserving JSONL."""

    if limit <= 0:
        raise ValueError("document limit must be positive")
    source = pq.ParquetFile(parquet_path)
    if "text" not in source.schema.names:
        raise ValueError(f"Parquet file has no text column: {source.schema.names}")

    destination = Path(jsonl_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as output:
        for batch in source.iter_batches(batch_size=2048, columns=["text"]):
            for text in batch.column(0).to_pylist():
                text = text.strip()
                if not text:
                    continue
                output.write(json.dumps({"id": count, "text": text}, ensure_ascii=False) + "\n")
                count += 1
                if count >= limit:
                    return count
    return count


def iter_jsonl_texts(path: str | Path) -> Iterator[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(f"invalid text at {path}:{line_number}")
            yield text


def tokenize_documents(
    documents_path: str | Path,
    output_path: str | Path,
    tokenizer: BPETokenizer,
) -> tuple[int, int]:
    """Write BOS/document/EOS sequences to a compact uint16 token stream."""

    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("vocabulary is too large for uint16 storage")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document_count = 0
    token_count = 0
    with destination.open("wb") as output:
        for text in iter_jsonl_texts(documents_path):
            token_ids = tokenizer.encode(text, add_document_tokens=True)
            np.asarray(token_ids, dtype=np.uint16).tofile(output)
            document_count += 1
            token_count += len(token_ids)
    return document_count, token_count


def prepare_tinystories(
    root: str | Path,
    train_documents: int = 50_000,
    validation_documents: int = 5_000,
    vocab_size: int = 8_000,
) -> dict[str, object]:
    """Prepare an official TinyStories subset for token-level model training."""

    root = Path(root)
    raw_dir = root / "raw"
    documents_dir = root / "documents"
    processed_dir = root / "processed"
    train_parquet = download_file(TRAIN_URL, raw_dir / "train-00000.parquet", TRAIN_SHA256)
    val_parquet = download_file(
        VALIDATION_URL, raw_dir / "validation-00000.parquet", VALIDATION_SHA256
    )

    train_jsonl = documents_dir / "train.jsonl"
    val_jsonl = documents_dir / "validation.jsonl"
    actual_train_documents = extract_documents(train_parquet, train_jsonl, train_documents)
    actual_val_documents = extract_documents(val_parquet, val_jsonl, validation_documents)
    print(f"Extracted {actual_train_documents:,} train and {actual_val_documents:,} validation documents")

    print(f"Training byte-level BPE tokenizer with target vocabulary {vocab_size:,}...")
    tokenizer = BPETokenizer.train(iter_jsonl_texts(train_jsonl), vocab_size=vocab_size)
    processed_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(processed_dir / "tokenizer.json")

    train_document_count, train_tokens = tokenize_documents(
        train_jsonl, processed_dir / "train.bin", tokenizer
    )
    val_document_count, val_tokens = tokenize_documents(
        val_jsonl, processed_dir / "val.bin", tokenizer
    )
    metadata: dict[str, object] = {
        "format": "uint16",
        "dataset": "roneneldan/TinyStories",
        "license": "cdla-sharing-1.0",
        "train_source": TRAIN_URL,
        "validation_source": VALIDATION_URL,
        "train_source_sha256": TRAIN_SHA256,
        "validation_source_sha256": VALIDATION_SHA256,
        "vocab_size": tokenizer.vocab_size,
        "train_documents": train_document_count,
        "val_documents": val_document_count,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
    }
    (processed_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
