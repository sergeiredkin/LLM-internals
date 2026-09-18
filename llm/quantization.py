"""Inference-only weight quantization primitives."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F


def quantize_int8_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically quantize each output row to signed INT8.

    Returns an INT8 matrix and FP32 scales with shape ``(out_features, 1)``. Zero rows use a
    scale of one and therefore reconstruct exactly as zero.
    """

    if weight.ndim != 2:
        raise ValueError("weight must have shape (out_features, in_features)")
    if not weight.is_floating_point():
        raise TypeError("weight must be floating point")
    weight_float = weight.detach().float()
    absmax = weight_float.abs().amax(dim=1, keepdim=True)
    scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
    quantized = torch.round(weight_float / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def dequantize_int8_per_channel(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct a floating-point weight matrix from per-channel INT8 values."""

    if quantized.ndim != 2 or quantized.dtype != torch.int8:
        raise ValueError("quantized weight must be a two-dimensional INT8 tensor")
    if scale.shape != (quantized.shape[0], 1) or not scale.is_floating_point():
        raise ValueError("scale must be floating point with shape (out_features, 1)")
    return quantized.to(dtype) * scale.to(dtype)


class Int8Linear(nn.Module):
    """Linear layer storing per-output-channel INT8 weights.

    This educational implementation dequantizes before ``F.linear``. It reduces persistent and
    serialized weight storage but does not provide an integer GEMM speedup.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight_q", torch.empty(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer("scale", torch.empty(out_features, 1, dtype=torch.float32))
        self.register_buffer(
            "bias", torch.empty(out_features, dtype=torch.float32) if bias else None
        )

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "Int8Linear":
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
        ).to(linear.weight.device)
        weight_q, scale = quantize_int8_per_channel(linear.weight)
        layer.weight_q.copy_(weight_q)
        layer.scale.copy_(scale)
        if linear.bias is not None:
            assert layer.bias is not None
            layer.bias.copy_(linear.bias.detach().float())
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_int8_per_channel(
            self.weight_q, self.scale, dtype=x.dtype
        )
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, per_channel=True"
        )


def replace_linear_with_int8(
    module: nn.Module,
    *,
    exclude: Iterable[str] = (),
) -> list[str]:
    """Recursively replace float linear layers and return their qualified names."""

    excluded = set(exclude)
    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            qualified_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and qualified_name not in excluded:
                setattr(parent, name, Int8Linear.from_float(child))
                replaced.append(qualified_name)
            else:
                visit(child, qualified_name)

    visit(module, "")
    return replaced


def module_storage_bytes(module: nn.Module) -> int:
    """Count unique parameter and buffer storage bytes, respecting tied tensors."""

    total = 0
    seen: set[tuple[torch.device, int]] = set()
    for tensor in list(module.parameters()) + list(module.buffers()):
        key = (tensor.device, tensor.untyped_storage().data_ptr())
        if key in seen:
            continue
        seen.add(key)
        total += tensor.untyped_storage().nbytes()
    return total
