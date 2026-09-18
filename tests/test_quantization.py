import io
import unittest

import torch
from torch import nn

from llm.config import ModelConfig
from llm.model import GPT
from llm.quantization import (
    Int8Linear,
    dequantize_int8_per_channel,
    module_storage_bytes,
    quantize_int8_per_channel,
    replace_linear_with_int8,
)


class Int8QuantizationTests(unittest.TestCase):
    def test_per_channel_quantization_shape_range_and_zero_row(self) -> None:
        weight = torch.tensor([[0.0, 0.0, 0.0], [-2.0, 0.5, 1.0]])
        weight_q, scale = quantize_int8_per_channel(weight)
        self.assertEqual(weight_q.dtype, torch.int8)
        self.assertEqual(weight_q.shape, weight.shape)
        self.assertEqual(scale.shape, (2, 1))
        self.assertLessEqual(weight_q.abs().max().item(), 127)
        reconstructed = dequantize_int8_per_channel(weight_q, scale)
        torch.testing.assert_close(reconstructed[0], weight[0])
        self.assertEqual(weight_q[1, 0].item(), -127)

    def test_per_channel_reconstruction_error_is_small(self) -> None:
        torch.manual_seed(4)
        weight = torch.randn(64, 128)
        weight_q, scale = quantize_int8_per_channel(weight)
        reconstructed = dequantize_int8_per_channel(weight_q, scale)
        relative_error = (reconstructed - weight).norm() / weight.norm()
        self.assertLess(relative_error.item(), 0.01)

    def test_int8_linear_approximates_float_linear(self) -> None:
        torch.manual_seed(5)
        linear = nn.Linear(32, 24, bias=True).eval()
        quantized = Int8Linear.from_float(linear).eval()
        inputs = torch.randn(3, 7, 32)
        expected = linear(inputs)
        actual = quantized(inputs)
        relative_error = (actual - expected).norm() / expected.norm()
        self.assertLess(relative_error.item(), 0.02)

    def test_replacement_can_exclude_a_qualified_layer(self) -> None:
        module = nn.Sequential(
            nn.Linear(8, 12),
            nn.Sequential(nn.Linear(12, 8), nn.Linear(8, 4)),
        )
        before = module_storage_bytes(module)
        replaced = replace_linear_with_int8(module, exclude={"1.1"})
        self.assertEqual(replaced, ["0", "1.0"])
        self.assertIsInstance(module[0], Int8Linear)
        self.assertIsInstance(module[1][0], Int8Linear)
        self.assertIsInstance(module[1][1], nn.Linear)
        self.assertLess(module_storage_bytes(module), before)

    def test_quantized_state_dict_round_trip(self) -> None:
        torch.manual_seed(6)
        source = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))
        replace_linear_with_int8(source)
        buffer = io.BytesIO()
        torch.save(source.state_dict(), buffer)

        target = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))
        replace_linear_with_int8(target)
        buffer.seek(0)
        target.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        inputs = torch.randn(2, 8)
        torch.testing.assert_close(target(inputs), source(inputs))

    def test_gpt_hidden_layers_quantize_without_breaking_tied_head(self) -> None:
        config = ModelConfig(
            vocab_size=32,
            context_length=16,
            n_layers=2,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            mlp_ratio=2.0,
            mlp_type="swiglu",
            position_encoding="rope",
            dropout=0.0,
        )
        model = GPT(config).eval()
        tokens = torch.randint(0, config.vocab_size, (2, 10))
        with torch.no_grad():
            expected = model(tokens)[0]
        replaced = replace_linear_with_int8(model, exclude={"lm_head"})
        with torch.no_grad():
            actual = model(tokens)[0]
        self.assertEqual(len(replaced), config.n_layers * 7)
        self.assertIsInstance(model.lm_head, nn.Linear)
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)
        relative_error = (actual - expected).norm() / expected.norm()
        self.assertLess(relative_error.item(), 0.05)

    def test_invalid_quantization_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            quantize_int8_per_channel(torch.randn(4))
        with self.assertRaisesRegex(TypeError, "floating"):
            quantize_int8_per_channel(torch.ones(2, 3, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "INT8"):
            dequantize_int8_per_channel(torch.ones(2, 3), torch.ones(2, 1))


if __name__ == "__main__":
    unittest.main()
