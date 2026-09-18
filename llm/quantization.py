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


def pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack pairs of signed INT4 values into uint8 nibbles."""

    if values.dtype != torch.int8:
        raise TypeError("INT4 values must use torch.int8 before packing")
    if values.ndim < 1:
        raise ValueError("INT4 values must have at least one dimension")
    if torch.any(values < -8) or torch.any(values > 7):
        raise ValueError("INT4 values must be in [-8, 7]")
    if values.shape[-1] % 2:
        values = F.pad(values, (0, 1))
    codes = values.to(torch.int16) & 0xF
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def unpack_signed_int4(packed: torch.Tensor, value_count: int) -> torch.Tensor:
    """Unpack uint8 nibbles into signed INT8 storage values."""

    if packed.dtype != torch.uint8 or packed.ndim < 1:
        raise ValueError("packed values must be a uint8 tensor")
    if value_count <= 0 or value_count > packed.shape[-1] * 2:
        raise ValueError("value_count is incompatible with packed width")
    low = (packed & 0xF).to(torch.int8)
    high = ((packed >> 4) & 0xF).to(torch.int8)
    values = torch.stack((low, high), dim=-1).flatten(-2)
    values = torch.where(values >= 8, values - 16, values)
    return values[..., :value_count]


def quantize_int4_groupwise(
    weight: torch.Tensor,
    group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each row in fixed-width groups and pack two values per byte."""

    if weight.ndim != 2:
        raise ValueError("weight must have shape (out_features, in_features)")
    if not weight.is_floating_point():
        raise TypeError("weight must be floating point")
    if group_size <= 0 or group_size % 2:
        raise ValueError("group_size must be a positive even integer")
    out_features, in_features = weight.shape
    group_count = (in_features + group_size - 1) // group_size
    padded_width = group_count * group_size
    weight_float = F.pad(weight.detach().float(), (0, padded_width - in_features))
    groups = weight_float.view(out_features, group_count, group_size)
    absmax = groups.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(absmax > 0, absmax / 7.0, torch.ones_like(absmax))
    quantized = torch.round(groups / scale).clamp(-7, 7).to(torch.int8)
    packed = pack_signed_int4(quantized.view(out_features, padded_width))
    return packed, scale.squeeze(-1)


def dequantize_int4_groupwise(
    packed: torch.Tensor,
    scale: torch.Tensor,
    in_features: int,
    group_size: int = 64,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unpack and reconstruct a groupwise INT4 weight matrix."""

    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("packed weight must be a two-dimensional uint8 tensor")
    if in_features <= 0 or group_size <= 0 or group_size % 2:
        raise ValueError("in_features and group_size must be positive; group_size must be even")
    group_count = (in_features + group_size - 1) // group_size
    padded_width = group_count * group_size
    if packed.shape[1] * 2 != padded_width:
        raise ValueError("packed width does not match in_features and group_size")
    if scale.shape != (packed.shape[0], group_count) or not scale.is_floating_point():
        raise ValueError("scale shape must be (out_features, group_count)")
    values = unpack_signed_int4(packed, padded_width).to(dtype)
    groups = values.view(packed.shape[0], group_count, group_size)
    reconstructed = groups * scale.to(dtype).unsqueeze(-1)
    return reconstructed.view(packed.shape[0], padded_width)[:, :in_features]


class Int4Linear(nn.Module):
    """Linear layer storing groupwise symmetric weights as packed INT4 nibbles."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 64,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if group_size <= 0 or group_size % 2:
            raise ValueError("group_size must be a positive even integer")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        group_count = (in_features + group_size - 1) // group_size
        padded_width = group_count * group_size
        self.register_buffer(
            "weight_packed",
            torch.empty(out_features, padded_width // 2, dtype=torch.uint8),
        )
        self.register_buffer(
            "scale", torch.empty(out_features, group_count, dtype=torch.float32)
        )
        self.register_buffer(
            "bias", torch.empty(out_features, dtype=torch.float32) if bias else None
        )

    @classmethod
    def from_float(cls, linear: nn.Linear, group_size: int = 64) -> "Int4Linear":
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
        ).to(linear.weight.device)
        packed, scale = quantize_int4_groupwise(linear.weight, group_size)
        layer.weight_packed.copy_(packed)
        layer.scale.copy_(scale)
        if linear.bias is not None:
            assert layer.bias is not None
            layer.bias.copy_(linear.bias.detach().float())
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_int4_groupwise(
            self.weight_packed,
            self.scale,
            self.in_features,
            self.group_size,
            dtype=x.dtype,
        )
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, group_size={self.group_size}"
        )


def replace_linear_with_int4(
    module: nn.Module,
    *,
    group_size: int = 64,
    exclude: Iterable[str] = (),
) -> list[str]:
    """Recursively replace float linear layers with packed groupwise INT4 layers."""

    excluded = set(exclude)
    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            qualified_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and qualified_name not in excluded:
                setattr(parent, name, Int4Linear.from_float(child, group_size))
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
