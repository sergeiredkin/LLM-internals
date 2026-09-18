"""Preallocated key/value caches for autoregressive inference."""

from __future__ import annotations

import torch


class LayerKVCache:
    """Fixed-capacity K/V storage for one transformer layer.

    Tensors use ``(batch, kv_heads, sequence, head_dim)``. GQA caches the
    unexpanded K/V heads, which is the source of its decode-memory savings.
    """

    def __init__(
        self,
        batch_size: int,
        n_kv_heads: int,
        max_length: int,
        head_dim: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        dimensions = (batch_size, n_kv_heads, max_length, head_dim)
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("cache dimensions must be positive")
        if not dtype.is_floating_point:
            raise TypeError("cache dtype must be floating point")

        self.batch_size = batch_size
        self.n_kv_heads = n_kv_heads
        self.max_length = max_length
        self.head_dim = head_dim
        self.keys = torch.empty(dimensions, device=device, dtype=dtype)
        self.values = torch.empty_like(self.keys)
        self.length = 0

    @property
    def remaining(self) -> int:
        return self.max_length - self.length

    @property
    def memory_bytes(self) -> int:
        return (
            self.keys.numel() * self.keys.element_size()
            + self.values.numel() * self.values.element_size()
        )

    def _validate(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.shape != values.shape:
            raise ValueError("keys and values must have identical shapes")
        if keys.ndim != 4:
            raise ValueError("keys and values must have shape (batch, kv_heads, sequence, head_dim)")
        expected = (self.batch_size, self.n_kv_heads, keys.shape[2], self.head_dim)
        if keys.shape != expected:
            raise ValueError(f"cache append expected shape {expected}, got {tuple(keys.shape)}")
        if keys.device != self.keys.device or values.device != self.values.device:
            raise ValueError("cache and appended tensors must use the same device")
        if keys.dtype != self.keys.dtype or values.dtype != self.values.dtype:
            raise ValueError("cache and appended tensors must use the same dtype")
        if keys.shape[2] == 0:
            raise ValueError("cannot append an empty sequence")
        if keys.shape[2] > self.remaining:
            raise ValueError(
                f"cache capacity exceeded: {keys.shape[2]} tokens requested, "
                f"{self.remaining} remaining"
            )

    def append(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append one or more positions and return the complete active cache."""

        self._validate(keys, values)
        new_length = self.length + keys.shape[2]
        self.keys[:, :, self.length : new_length].copy_(keys)
        self.values[:, :, self.length : new_length].copy_(values)
        self.length = new_length
        return self.active()

    def active(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return views containing only populated positions."""

        return self.keys[:, :, : self.length], self.values[:, :, : self.length]

    def reset(self) -> None:
        """Mark the cache empty while retaining its allocated storage."""

        self.length = 0
