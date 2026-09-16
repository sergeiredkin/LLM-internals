"""Small, inspectable tokenizers used by the educational training pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch


class CharTokenizer:
    """Deterministic character tokenizer with a sorted vocabulary."""

    def __init__(self, vocabulary: Iterable[str]) -> None:
        self.tokens = tuple(vocabulary)
        if not self.tokens:
            raise ValueError("tokenizer vocabulary cannot be empty")
        if any(len(token) != 1 for token in self.tokens):
            raise ValueError("every character token must contain exactly one character")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("tokenizer vocabulary contains duplicates")
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        if not text:
            raise ValueError("cannot build a tokenizer from empty text")
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.token_to_id[character] for character in text]
        except KeyError as error:
            character = error.args[0]
            raise ValueError(f"character {character!r} is not in the vocabulary") from error

    def decode(self, token_ids: Iterable[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().flatten().tolist()
        output: list[str] = []
        for token_id in token_ids:
            if not 0 <= int(token_id) < self.vocab_size:
                raise ValueError(f"token ID {token_id} is outside the vocabulary")
            output.append(self.tokens[int(token_id)])
        return "".join(output)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"type": "char", "tokens": list(self.tokens)}
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "char" or not isinstance(payload.get("tokens"), list):
            raise ValueError("invalid character tokenizer file")
        return cls(payload["tokens"])
