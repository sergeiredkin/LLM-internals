"""Corpus preparation and random next-token batching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .tokenizer import CharTokenizer


Split = Literal["train", "val"]


def prepare_char_corpus(
    input_path: str | Path,
    output_dir: str | Path,
    train_fraction: float = 0.9,
) -> dict[str, object]:
    """Tokenize UTF-8 text and write compact uint16 train/validation files."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    source = Path(input_path)
    text = source.read_text(encoding="utf-8")
    if len(text) < 2:
        raise ValueError("corpus must contain at least two characters")

    tokenizer = CharTokenizer.from_text(text)
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("character vocabulary is too large for uint16 storage")

    token_ids = np.asarray(tokenizer.encode(text), dtype=np.uint16)
    split_index = int(len(token_ids) * train_fraction)
    if split_index < 2 or len(token_ids) - split_index < 2:
        raise ValueError("train and validation splits must each contain at least two tokens")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train_tokens = token_ids[:split_index]
    val_tokens = token_ids[split_index:]
    train_tokens.tofile(destination / "train.bin")
    val_tokens.tofile(destination / "val.bin")
    tokenizer.save(destination / "tokenizer.json")

    metadata: dict[str, object] = {
        "format": "uint16",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "vocab_size": tokenizer.vocab_size,
        "total_tokens": len(token_ids),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_fraction": train_fraction,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


class TokenCorpus:
    """Memory-mapped train/validation token streams."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        metadata_path = self.directory / "metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("format") != "uint16":
            raise ValueError("unsupported token storage format")

        self.tokenizer = CharTokenizer.load(self.directory / "tokenizer.json")
        if self.tokenizer.vocab_size != self.metadata.get("vocab_size"):
            raise ValueError("tokenizer and corpus metadata vocabularies disagree")

        self.train = np.memmap(self.directory / "train.bin", dtype=np.uint16, mode="r")
        self.val = np.memmap(self.directory / "val.bin", dtype=np.uint16, mode="r")
        if len(self.train) != self.metadata.get("train_tokens"):
            raise ValueError("train token count does not match metadata")
        if len(self.val) != self.metadata.get("val_tokens"):
            raise ValueError("validation token count does not match metadata")

    def get_batch(
        self,
        split: Split,
        batch_size: int,
        context_length: int,
        device: str | torch.device = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample input windows and their one-token-shifted targets."""

        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if batch_size <= 0 or context_length <= 0:
            raise ValueError("batch_size and context_length must be positive")

        data = self.train if split == "train" else self.val
        max_start = len(data) - context_length - 1
        if max_start < 0:
            raise ValueError(
                f"{split} split has {len(data)} tokens, fewer than the requested "
                f"context length {context_length} plus one target"
            )

        starts = torch.randint(
            0, max_start + 1, (batch_size,), generator=generator
        ).tolist()
        inputs = torch.stack(
            [
                torch.from_numpy(
                    np.array(data[start : start + context_length], dtype=np.int64)
                )
                for start in starts
            ]
        )
        targets = torch.stack(
            [
                torch.from_numpy(
                    np.array(data[start + 1 : start + context_length + 1], dtype=np.int64)
                )
                for start in starts
            ]
        )

        target_device = torch.device(device)
        if target_device.type == "cuda":
            inputs = inputs.pin_memory()
            targets = targets.pin_memory()
        return (
            inputs.to(target_device, non_blocking=target_device.type == "cuda"),
            targets.to(target_device, non_blocking=target_device.type == "cuda"),
        )
