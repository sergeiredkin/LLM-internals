#!/usr/bin/env python3
"""Download and prepare a document-preserving TinyStories BPE corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm.tinystories import prepare_tinystories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/tinystories"))
    parser.add_argument("--train-documents", type=int, default=50_000)
    parser.add_argument("--validation-documents", type=int, default=5_000)
    parser.add_argument("--vocab-size", type=int, default=8_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_tinystories(
        args.root,
        train_documents=args.train_documents,
        validation_documents=args.validation_documents,
        vocab_size=args.vocab_size,
    )
    print("\nTinyStories preparation complete")
    print(f"Vocabulary:           {metadata['vocab_size']:,}")
    print(f"Training documents:   {metadata['train_documents']:,}")
    print(f"Validation documents: {metadata['val_documents']:,}")
    print(f"Training tokens:      {metadata['train_tokens']:,}")
    print(f"Validation tokens:    {metadata['val_tokens']:,}")


if __name__ == "__main__":
    main()
