"""Optimization, scheduling, precision, and evaluation utilities."""

from __future__ import annotations

import contextlib
import math

import torch
from torch import nn

from .config import TrainingConfig
from .data import TokenCorpus


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def autocast_context(
    device: torch.device, precision: str
) -> contextlib.AbstractContextManager[None]:
    if precision == "fp32":
        return contextlib.nullcontext()
    if device.type == "cpu":
        if precision == "bf16":
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        # CPU fp16 support is incomplete and generally slower than fp32.
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def learning_rate_at(step: int, config: TrainingConfig) -> float:
    """Linear warmup followed by cosine decay to min_learning_rate."""

    if step < 0:
        raise ValueError("step cannot be negative")
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    if step >= config.max_steps:
        return config.min_learning_rate

    decay_steps = max(config.max_steps - config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / decay_steps
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + coefficient * (
        config.learning_rate - config.min_learning_rate
    )


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    """Apply weight decay to matrix weights, not norms or biases."""

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    decay = [parameter for parameter in trainable if parameter.ndim >= 2]
    no_decay = [parameter for parameter in trainable if parameter.ndim < 2]
    if len({id(parameter) for parameter in decay + no_decay}) != len(trainable):
        raise RuntimeError("optimizer parameter groups overlap or omit parameters")

    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )


def create_grad_scaler(device: torch.device, precision: str) -> torch.amp.GradScaler:
    # BF16 has enough exponent range not to require dynamic loss scaling.
    enabled = device.type == "cuda" and precision == "fp16"
    return torch.amp.GradScaler("cuda", enabled=enabled)


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    corpus: TokenCorpus,
    split: str,
    batches: int,
    batch_size: int,
    context_length: int,
    device: torch.device,
    precision: str,
    generator: torch.Generator | None = None,
) -> float:
    """Estimate mean next-token loss without changing the model's mode."""

    if batches <= 0:
        raise ValueError("batches must be positive")
    was_training = model.training
    model.eval()
    losses = torch.empty(batches, dtype=torch.float64)
    try:
        for index in range(batches):
            inputs, targets = corpus.get_batch(
                split, batch_size, context_length, device, generator
            )
            with autocast_context(device, precision):
                _, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("model did not return a loss")
            losses[index] = loss.detach().float().cpu()
    finally:
        model.train(was_training)
    return losses.mean().item()


def train_micro_batches(
    model: nn.Module,
    corpus: TokenCorpus,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: TrainingConfig,
    context_length: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[float, float]:
    """Perform one optimizer update and return unscaled loss and gradient norm."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    for _ in range(config.gradient_accumulation_steps):
        inputs, targets = corpus.get_batch(
            "train", config.micro_batch_size, context_length, device, generator
        )
        with autocast_context(device, config.precision):
            _, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("model did not return a loss")
            scaled_loss = loss / config.gradient_accumulation_steps
        accumulated_loss += loss.detach().float().item()
        scaler.scale(scaled_loss).backward()

    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return (
        accumulated_loss / config.gradient_accumulation_steps,
        float(gradient_norm.detach().float().cpu()),
    )
