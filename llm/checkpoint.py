"""Atomic, resumable training checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import ExperimentConfig


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: ExperimentConfig,
    step: int,
    best_val_loss: float,
) -> None:
    """Write a checkpoint atomically so interruption cannot corrupt the last file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    state: dict[str, Any] = {
        "format_version": 1,
        "step": step,
        "best_val_loss": best_val_loss,
        "config": config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, temporary)
    os.replace(temporary, destination)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load model/training state and return checkpoint metadata."""

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if restore_rng:
        torch.set_rng_state(checkpoint["cpu_rng_state"])
        cuda_states = checkpoint.get("cuda_rng_states")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_states)
    return checkpoint
