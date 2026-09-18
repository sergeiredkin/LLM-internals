"""Low-rank adapters for parameter-efficient fine-tuning."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .quantization import Int4Linear


class LoRALinear(nn.Module):
    """Wrap a frozen linear-like base with a trainable low-rank update."""

    def __init__(
        self,
        base: nn.Module,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.base = base
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear | Int4Linear,
        *,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        return cls(
            linear,
            linear.in_features,
            linear.out_features,
            rank,
            alpha,
            dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        adapter = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_output + adapter.to(base_output.dtype) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        """Return the floating-point `(out_features, in_features)` update."""

        return (self.lora_B @ self.lora_A) * self.scaling

    def merge(self) -> nn.Linear:
        """Merge into a floating-point base and return the ordinary linear layer."""

        if not isinstance(self.base, nn.Linear):
            raise TypeError("only floating-point nn.Linear bases can be merged")
        with torch.no_grad():
            self.base.weight.add_(self.delta_weight().to(self.base.weight.dtype))
        return self.base

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha:g}, dropout={self.dropout.p}"
        )


def apply_lora(
    module: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Iterable[str] = ("q_proj", "v_proj"),
    quantized_base: bool = False,
) -> list[str]:
    """Freeze a model and wrap matching linear module names.

    Set ``quantized_base=True`` to include packed ``Int4Linear`` modules for QLoRA.
    """

    targets = set(target_modules)
    if not targets:
        raise ValueError("target_modules cannot be empty")
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            qualified_name = f"{prefix}.{name}" if prefix else name
            supported = isinstance(child, nn.Linear) or (
                quantized_base and isinstance(child, Int4Linear)
            )
            if supported and (name in targets or qualified_name in targets):
                setattr(
                    parent,
                    name,
                    LoRALinear.from_linear(
                        child,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    ),
                )
                replaced.append(qualified_name)
            else:
                visit(child, qualified_name)

    visit(module, "")
    if not replaced:
        raise ValueError("no linear modules matched target_modules")
    return replaced


def merge_lora(module: nn.Module) -> list[str]:
    """Merge every floating-point LoRA wrapper in place."""

    merged: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            qualified_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merge())
                merged.append(qualified_name)
            else:
                visit(child, qualified_name)

    visit(module, "")
    return merged


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return only adapter tensors, detached on CPU for compact serialization."""

    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
        if name.endswith("lora_A") or name.endswith("lora_B")
    }


def load_lora_state_dict(
    module: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    """Strictly load an adapter-only state dictionary."""

    expected = set(lora_state_dict(module))
    provided = set(state)
    if expected != provided:
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        raise ValueError(f"LoRA state mismatch: missing={missing}, unexpected={unexpected}")
    parameters = dict(module.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            target = parameters[name]
            if target.shape != value.shape:
                raise ValueError(
                    f"LoRA tensor {name} has shape {tuple(value.shape)}, "
                    f"expected {tuple(target.shape)}"
                )
            target.copy_(value.to(device=target.device, dtype=target.dtype))


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
