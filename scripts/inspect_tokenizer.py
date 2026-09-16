#!/usr/bin/env python3
"""Inspect how the TinyStories BPE tokenizer splits text."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm.tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tinystories/processed/tokenizer.json"),
    )
    parser.add_argument("--text", help="text to tokenize; prompts interactively when omitted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.text if args.text is not None else input("Text: ")
    if not text:
        raise SystemExit("Text cannot be empty")

    tokenizer = BPETokenizer.load(args.tokenizer)
    encoding = tokenizer.tokenizer.encode(text)
    print(f"Vocabulary: {tokenizer.vocab_size:,}")
    print(f"Characters: {len(text):,} | BPE tokens: {len(encoding.ids):,}")
    print(f"Compression: {len(text) / max(len(encoding.ids), 1):.2f} characters/token")
    print("\nTokens:")
    for index, (token_id, token, offsets) in enumerate(
        zip(encoding.ids, encoding.tokens, encoding.offsets)
    ):
        print(f"  {index:3d} id={token_id:4d} token={token!r:18s} source={text[offsets[0]:offsets[1]]!r}")
    print(f"\nRound trip: {tokenizer.decode(encoding.ids)!r}")


if __name__ == "__main__":
    main()
