"""Small, inspectable tokenizers used by the educational training pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


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
            output.append(self.token_for_id(int(token_id)))
        return "".join(output)

    def token_for_id(self, token_id: int) -> str:
        if not 0 <= token_id < self.vocab_size:
            raise ValueError(f"token ID {token_id} is outside the vocabulary")
        return self.tokens[token_id]

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


class BPETokenizer:
    """Byte-level BPE tokenizer with explicit document boundary tokens."""

    SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        missing = [token for token in self.SPECIAL_TOKENS if tokenizer.token_to_id(token) is None]
        if missing:
            raise ValueError(f"BPE tokenizer is missing special tokens: {missing}")

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 8000) -> "BPETokenizer":
        if vocab_size <= len(cls.SPECIAL_TOKENS):
            raise ValueError("BPE vocab_size is too small")
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=list(cls.SPECIAL_TOKENS),
            show_progress=True,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return cls(tokenizer)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def bos_id(self) -> int:
        token_id = self.tokenizer.token_to_id("<bos>")
        assert token_id is not None
        return token_id

    @property
    def eos_id(self) -> int:
        token_id = self.tokenizer.token_to_id("<eos>")
        assert token_id is not None
        return token_id

    def encode(self, text: str, add_document_tokens: bool = False) -> list[int]:
        token_ids = self.tokenizer.encode(text).ids
        if add_document_tokens:
            return [self.bos_id, *token_ids, self.eos_id]
        return token_ids

    def decode(self, token_ids: Iterable[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().flatten().tolist()
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def token_for_id(self, token_id: int) -> str:
        token = self.tokenizer.id_to_token(token_id)
        if token is None:
            raise ValueError(f"token ID {token_id} is outside the vocabulary")
        return token

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(destination))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls(Tokenizer.from_file(str(path)))


def load_tokenizer(path: str | Path) -> CharTokenizer | BPETokenizer:
    """Load either project tokenizer format from its JSON file."""

    tokenizer_path = Path(path)
    payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    if payload.get("type") == "char":
        return CharTokenizer.load(tokenizer_path)
    if "model" in payload:
        return BPETokenizer.load(tokenizer_path)
    raise ValueError("unrecognized tokenizer format")
