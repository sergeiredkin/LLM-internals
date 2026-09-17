"""Causal self-attention implemented with PyTorch SDPA."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .rope import RotaryEmbedding


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    """Expand shared K/V heads to match query heads for standard SDPA.

    For two KV heads and four repeats, head order becomes
    ``[kv0, kv0, kv0, kv0, kv1, kv1, kv1, kv1]``.
    """

    if x.ndim != 4:
        raise ValueError("K/V tensor must have shape (batch, heads, sequence, head_dim)")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if repeats == 1:
        return x
    return x.repeat_interleave(repeats, dim=1)


class CausalSelfAttention(nn.Module):
    """Multi-head/GQA causal attention without an inference cache yet."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout
        self.rope = (
            RotaryEmbedding(self.head_dim)
            if config.position_encoding == "rope"
            else None
        )

        kv_width = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, kv_width, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, kv_width, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _split_heads(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        batch, sequence, _ = x.shape
        return x.view(batch, sequence, n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        q = self._split_heads(self.q_proj(x), self.n_heads)
        k = self._split_heads(self.k_proj(x), self.n_kv_heads)
        v = self._split_heads(self.v_proj(x), self.n_kv_heads)

        if self.rope is not None:
            q, k = self.rope(q, k)

        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = repeat_kv(k, repeats)
            v = repeat_kv(v, repeats)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        output = output.transpose(1, 2).contiguous().view(batch, sequence, -1)
        return self.resid_dropout(self.out_proj(output))
