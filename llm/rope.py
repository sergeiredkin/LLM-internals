"""Rotary position embeddings, kept independent for testing and study."""

from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate every adjacent coordinate pair by 90 degrees.

    ``(x0, x1)`` becomes ``(-x1, x0)``. RoPE combines this helper with
    cosine and sine weights to produce arbitrary rotation angles.
    """

    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE requires an even final dimension")
    pairs = x.unflatten(-1, (-1, 2))
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1)
    return rotated.flatten(-2)


class RotaryEmbedding(nn.Module):
    """Apply fixed position-dependent rotations to query and key tensors.

    Inputs use ``(batch, heads, sequence, head_dim)``. Query and key may
    have different head counts, which keeps this component compatible with
    grouped-query attention later.
    """

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")
        if base <= 0:
            raise ValueError("base must be positive")

        self.head_dim = head_dim
        self.base = float(base)
        pair_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inverse_frequencies = self.base ** (-pair_indices / head_dim)
        self.register_buffer("inverse_frequencies", inverse_frequencies, persistent=False)
        self.register_buffer("_cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("_sin_cached", torch.empty(0), persistent=False)

    def _cos_sin(
        self,
        required_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_is_valid = (
            self._cos_cached.shape[0] >= required_length
            and self._cos_cached.device == device
            and self._cos_cached.dtype == dtype
        )
        if not cache_is_valid:
            positions = torch.arange(required_length, device=device, dtype=torch.float32)
            # Compute angles in FP32 even when the model itself uses BF16.
            frequencies = self.inverse_frequencies.to(device=device, dtype=torch.float32)
            angles = torch.outer(positions, frequencies)
            # Repeat each angle for its adjacent coordinate pair.
            angles = angles.repeat_interleave(2, dim=-1)
            self._cos_cached = angles.cos().to(dtype=dtype)
            self._sin_cached = angles.sin().to(dtype=dtype)
        return self._cos_cached, self._sin_cached

    def _validate(self, q: torch.Tensor, k: torch.Tensor) -> None:
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError("q and k must have shape (batch, heads, sequence, head_dim)")
        if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
            raise ValueError("q and k must have matching batch and sequence dimensions")
        if q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError(f"q and k final dimension must equal head_dim={self.head_dim}")
        if q.device != k.device or q.dtype != k.dtype:
            raise ValueError("q and k must have matching devices and dtypes")
        if not q.is_floating_point():
            raise TypeError("q and k must be floating-point tensors")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate Q and K, optionally starting after a cached prefix."""

        self._validate(q, k)
        if position_offset < 0:
            raise ValueError("position_offset cannot be negative")

        sequence = q.shape[2]
        cos, sin = self._cos_sin(position_offset + sequence, q.device, q.dtype)
        cos = cos[position_offset : position_offset + sequence][None, None, :, :]
        sin = sin[position_offset : position_offset + sequence][None, None, :, :]
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
