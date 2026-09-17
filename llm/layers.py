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


def matched_swiglu_hidden_size(
    d_model: int,
    gelu_ratio: float = 4.0,
    multiple_of: int = 16,
) -> int:
    """Choose a hardware-friendly SwiGLU width near a parameter-matched GELU MLP.

    A bias-free GELU MLP has two matrices and a SwiGLU MLP has three, so
    matching parameters requires two thirds of the GELU hidden width.
    """

    if d_model <= 0 or gelu_ratio <= 0 or multiple_of <= 0:
        raise ValueError("d_model, gelu_ratio, and multiple_of must be positive")
    ideal = 2.0 * d_model * gelu_ratio / 3.0
    return max(multiple_of, round(ideal / multiple_of) * multiple_of)


class SwiGLU(nn.Module):
    """Gated feed-forward layer: down(SiLU(gate(x)) * value(x))."""

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0 or hidden_size <= 0:
            raise ValueError("d_model and hidden_size must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.gate_proj = nn.Linear(d_model, hidden_size, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden_size, bias=bias)
        self.down_proj = nn.Linear(hidden_size, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))


class MLP(nn.Module):
    """GELU feed-forward network used by the baseline GPT."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.mlp_hidden_size
        self.up_proj = nn.Linear(config.d_model, hidden, bias=config.bias)
        self.down_proj = nn.Linear(hidden, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.up_proj(x), approximate="tanh")
        return self.dropout(self.down_proj(x))
