"""Validated experiment configuration for training and generating with GPT models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelConfig:
    """Architecture settings shared by training and inference."""

    vocab_size: int | None = None
    context_length: int = 256
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    n_kv_heads: int = 6
    mlp_ratio: float = 4.0
    mlp_type: str = "gelu"  # gelu | swiglu
    swiglu_multiple_of: int = 16
    dropout: float = 0.1
    norm_eps: float = 1e-5
    bias: bool = False
    tie_embeddings: bool = True
    position_encoding: str = "learned"  # learned | rope

    def validate(self) -> None:
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError("model.vocab_size must be positive or null")
        for name in ("context_length", "n_layers", "d_model", "n_heads", "n_kv_heads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"model.{name} must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("model.d_model must be divisible by model.n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("model.n_heads must be divisible by model.n_kv_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("model.mlp_ratio must be positive")
        if self.mlp_type not in {"gelu", "swiglu"}:
            raise ValueError("model.mlp_type must be 'gelu' or 'swiglu'")
        if self.swiglu_multiple_of <= 0:
            raise ValueError("model.swiglu_multiple_of must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.norm_eps <= 0:
            raise ValueError("model.norm_eps must be positive")
        if self.position_encoding not in {"learned", "rope"}:
            raise ValueError("model.position_encoding must be 'learned' or 'rope'")
        if self.position_encoding == "rope" and self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even model head dimension")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def mlp_hidden_size(self) -> int:
        return int(self.d_model * self.mlp_ratio)


@dataclass(slots=True)
class DataConfig:
    """Corpus and tokenization settings."""

    input_path: str = "data/shakespeare/input.txt"
    processed_dir: str = "data/shakespeare/processed"
    tokenizer: str = "char"
    train_fraction: float = 0.9
    num_workers: int = 0

    def validate(self) -> None:
        if self.tokenizer not in {"char", "bpe"}:
            raise ValueError("data.tokenizer must be 'char' or 'bpe'")
        if not 0 < self.train_fraction < 1:
            raise ValueError("data.train_fraction must be between 0 and 1")
        if self.num_workers < 0:
            raise ValueError("data.num_workers cannot be negative")


@dataclass(slots=True)
class TrainingConfig:
    """Optimization, evaluation, and checkpoint settings."""

    seed: int = 1337
    device: str = "auto"
    precision: str = "bf16"  # fp32 | fp16 | bf16
    micro_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_steps: int = 5000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 50
    log_interval: int = 10
    checkpoint_interval: int = 500
    step_idle_seconds: float = 0.0
    output_dir: str = "runs/shakespeare"

    def validate(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("training.device must be 'auto', 'cpu', or 'cuda'")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("training.precision must be fp32, fp16, or bf16")
        positive_ints = (
            "micro_batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "eval_interval",
            "eval_batches",
            "log_interval",
            "checkpoint_interval",
        )
        for name in positive_ints:
            if getattr(self, name) <= 0:
                raise ValueError(f"training.{name} must be positive")
        if self.warmup_steps < 0 or self.warmup_steps >= self.max_steps:
            raise ValueError("training.warmup_steps must be in [0, max_steps)")
        if not 0 < self.min_learning_rate <= self.learning_rate:
            raise ValueError(
                "training.min_learning_rate must be positive and no greater than learning_rate"
            )
        if self.weight_decay < 0:
            raise ValueError("training.weight_decay cannot be negative")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("training beta values must be in [0, 1)")
        if self.grad_clip <= 0:
            raise ValueError("training.grad_clip must be positive")
        if self.step_idle_seconds < 0:
            raise ValueError("training.step_idle_seconds cannot be negative")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


@dataclass(slots=True)
class ExperimentConfig:
    """Complete configuration for one reproducible experiment."""

    name: str = "shakespeare"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("experiment name cannot be empty")
        self.model.validate()
        self.data.validate()
        self.training.validate()

    @property
    def tokens_per_update(self) -> int:
        return (
            self.training.effective_batch_size
            * self.model.context_length
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    """Construct and validate an experiment configuration from a mapping."""

    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    allowed = {"name", "model", "data", "training"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown top-level configuration keys: {sorted(unknown)}")

    config = ExperimentConfig(
        name=raw.get("name", "shakespeare"),
        model=ModelConfig(**(raw.get("model") or {})),
        data=DataConfig(**(raw.get("data") or {})),
        training=TrainingConfig(**(raw.get("training") or {})),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_dict(raw)
