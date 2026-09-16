#!/usr/bin/env python3
"""Prepare a configured text corpus for language-model training."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm.config import load_config
from llm.data import prepare_char_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/shakespeare.yaml"),
        help="experiment YAML file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.data.tokenizer != "char":
        raise SystemExit("Only the character tokenizer is implemented at this stage.")

    metadata = prepare_char_corpus(
        config.data.input_path,
        config.data.processed_dir,
        config.data.train_fraction,
    )
    print(f"Prepared:       {config.data.processed_dir}")
    print(f"Vocabulary:     {metadata['vocab_size']} characters")
    print(f"Training:       {metadata['train_tokens']:,} tokens")
    print(f"Validation:     {metadata['val_tokens']:,} tokens")
    print(f"Source SHA-256: {metadata['source_sha256']}")


if __name__ == "__main__":
    main()
