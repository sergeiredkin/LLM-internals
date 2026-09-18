import unittest

import torch
from torch import nn

from llm.config import ModelConfig
from llm.lora import (
    LoRALinear,
    apply_lora,
    load_lora_state_dict,
    lora_state_dict,
    merge_lora,
    trainable_parameter_count,
)
from llm.model import GPT
from llm.quantization import Int4Linear, replace_linear_with_int4


class LoRATests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(12)

    def test_zero_initialized_adapter_preserves_base_output_exactly(self) -> None:
        base = nn.Linear(16, 12, bias=True)
        inputs = torch.randn(3, 5, 16)
        expected = base(inputs)
        lora = LoRALinear.from_linear(base, rank=4, alpha=8).eval()
        torch.testing.assert_close(lora(inputs), expected, rtol=0, atol=0)
        self.assertTrue(torch.count_nonzero(lora.lora_A))
        self.assertEqual(torch.count_nonzero(lora.lora_B).item(), 0)

    def test_only_adapter_parameters_receive_gradients(self) -> None:
        lora = LoRALinear.from_linear(nn.Linear(8, 6), rank=2, alpha=4)
        loss = lora(torch.randn(4, 8)).square().mean()
        loss.backward()
        self.assertIsNotNone(lora.lora_A.grad)
        self.assertIsNotNone(lora.lora_B.grad)
        self.assertTrue(all(parameter.grad is None for parameter in lora.base.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in lora.base.parameters()))

    def test_delta_rank_cannot_exceed_adapter_rank(self) -> None:
        lora = LoRALinear.from_linear(nn.Linear(10, 9), rank=3, alpha=6)
        with torch.no_grad():
            lora.lora_B.normal_()
        numerical_rank = torch.linalg.matrix_rank(lora.delta_weight()).item()
        self.assertLessEqual(numerical_rank, 3)

    def test_merge_matches_unmerged_output(self) -> None:
        module = nn.Sequential(nn.Linear(10, 9), nn.ReLU(), nn.Linear(9, 5)).eval()
        apply_lora(module, rank=3, alpha=6, target_modules={"0", "2"})
        with torch.no_grad():
            for child in module.modules():
                if isinstance(child, LoRALinear):
                    child.lora_B.normal_(std=0.1)
        inputs = torch.randn(4, 10)
        expected = module(inputs)
        merged = merge_lora(module)
        actual = module(inputs)
        self.assertEqual(merged, ["0", "2"])
        self.assertFalse(any(isinstance(child, LoRALinear) for child in module.modules()))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_adapter_only_state_round_trip_is_strict(self) -> None:
        source = nn.Sequential(nn.Linear(8, 7), nn.Linear(7, 4))
        target = nn.Sequential(nn.Linear(8, 7), nn.Linear(7, 4))
        apply_lora(source, rank=2, alpha=4, target_modules={"0", "1"})
        apply_lora(target, rank=2, alpha=4, target_modules={"0", "1"})
        with torch.no_grad():
            for parameter in source.parameters():
                if parameter.requires_grad:
                    parameter.normal_()
        state = lora_state_dict(source)
        load_lora_state_dict(target, state)
        for name, value in lora_state_dict(target).items():
            torch.testing.assert_close(value, state[name])
        with self.assertRaisesRegex(ValueError, "mismatch"):
            load_lora_state_dict(target, {})

    def test_gpt_qv_lora_trainable_parameter_count(self) -> None:
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
        replaced = apply_lora(model, rank=4, alpha=8)
        with torch.no_grad():
            actual = model(tokens)[0]
        self.assertEqual(len(replaced), config.n_layers * 2)
        expected_trainable = config.n_layers * 4 * (
            config.d_model + config.d_model
            + config.d_model + config.n_kv_heads * config.head_dim
        )
        self.assertEqual(trainable_parameter_count(model), expected_trainable)
        self.assertTrue(all("lora_" in name for name, p in model.named_parameters() if p.requires_grad))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_qlora_wraps_frozen_int4_base_without_changing_initial_output(self) -> None:
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
        replace_linear_with_int4(model, group_size=16, exclude={"lm_head"})
        tokens = torch.randint(0, config.vocab_size, (2, 10))
        with torch.no_grad():
            expected = model(tokens)[0]
        replaced = apply_lora(
            model,
            rank=4,
            alpha=8,
            target_modules=("q_proj", "v_proj"),
            quantized_base=True,
        )
        with torch.no_grad():
            actual = model(tokens)[0]
        self.assertEqual(len(replaced), config.n_layers * 2)
        self.assertIsInstance(model.blocks[0].attn.q_proj.base, Int4Linear)
        self.assertEqual(trainable_parameter_count(model), 896)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        loss = model(tokens, tokens)[1]
        assert loss is not None
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )
        with self.assertRaisesRegex(TypeError, "floating-point"):
            merge_lora(model)

    def test_invalid_configuration_and_empty_targets_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank"):
            LoRALinear.from_linear(nn.Linear(4, 4), rank=0)
        with self.assertRaisesRegex(ValueError, "no linear"):
            apply_lora(nn.Linear(4, 4), target_modules={"missing"})


if __name__ == "__main__":
    unittest.main()
