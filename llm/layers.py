"""Reusable neural-network layers for the GPT model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """RMSNorm with an fp32 reduction for mixed-precision stability."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        normalized = x_float * torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return self.weight * normalized.to(input_dtype)


class MLP(nn.Module):
    """The initial GPT feed-forward network; SwiGLU will be added later."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.mlp_hidden_size
        self.up_proj = nn.Linear(config.d_model, hidden, bias=config.bias)
        self.down_proj = nn.Linear(hidden, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.up_proj(x), approximate="tanh")
        return self.dropout(self.down_proj(x))
