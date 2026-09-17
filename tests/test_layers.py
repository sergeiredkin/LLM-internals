from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from llm.config import ModelConfig
from llm.layers import MLP, SwiGLU, matched_swiglu_hidden_size


class SwiGLUTests(unittest.TestCase):
    def test_formula_with_identity_projections(self) -> None:
        layer = SwiGLU(d_model=2, hidden_size=2, bias=False)
        identity = torch.eye(2)
        with torch.no_grad():
            layer.gate_proj.weight.copy_(identity)
            layer.up_proj.weight.copy_(identity)
            layer.down_proj.weight.copy_(identity)
        x = torch.tensor([[[1.0, -2.0]]])
        expected = F.silu(x) * x
        torch.testing.assert_close(layer(x), expected)

    def test_output_shape_matches_input_shape(self) -> None:
        layer = SwiGLU(d_model=32, hidden_size=80)
        x = torch.randn(3, 7, 32)
        self.assertEqual(layer(x).shape, x.shape)

    def test_hidden_width_nearly_matches_gelu_parameter_count(self) -> None:
        config = ModelConfig(d_model=512, n_heads=8, n_kv_heads=8, mlp_ratio=4.0)
        gelu = MLP(config)
        hidden = matched_swiglu_hidden_size(config.d_model, config.mlp_ratio)
        swiglu = SwiGLU(config.d_model, hidden)
        gelu_parameters = sum(parameter.numel() for parameter in gelu.parameters())
        swiglu_parameters = sum(parameter.numel() for parameter in swiglu.parameters())
        self.assertEqual(hidden, 1360)
        self.assertLess(abs(swiglu_parameters - gelu_parameters) / gelu_parameters, 0.005)

    def test_bfloat16_dtype_and_finite_gradients(self) -> None:
        layer = SwiGLU(d_model=32, hidden_size=64).to(dtype=torch.bfloat16)
        x = torch.randn(2, 5, 32, dtype=torch.bfloat16, requires_grad=True)
        output = layer(x)
        self.assertEqual(output.dtype, torch.bfloat16)
        output.float().square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in layer.parameters()
            )
        )

    def test_invalid_settings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            SwiGLU(d_model=0, hidden_size=32)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
            SwiGLU(d_model=32, hidden_size=64, dropout=1.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            matched_swiglu_hidden_size(32, multiple_of=0)


if __name__ == "__main__":
    unittest.main()
