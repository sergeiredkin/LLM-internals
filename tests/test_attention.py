from __future__ import annotations

import unittest

import torch

from llm.attention import CausalSelfAttention, repeat_kv
from llm.cache import LayerKVCache
from llm.config import ModelConfig
from scripts.overfit_batch import small_attention_heads


def attention_config(n_kv_heads: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        context_length=16,
        n_layers=1,
        d_model=64,
        n_heads=8,
        n_kv_heads=n_kv_heads,
        dropout=0.0,
        position_encoding="rope",
    )


class GroupedQueryAttentionTests(unittest.TestCase):
    def test_repeat_kv_assigns_each_query_group_to_one_kv_head(self) -> None:
        # Two easily identified KV heads, each shared by four query heads.
        kv = torch.tensor([[[[10.0]], [[20.0]]]])
        expanded = repeat_kv(kv, repeats=4)
        self.assertEqual(expanded.shape, (1, 8, 1, 1))
        torch.testing.assert_close(
            expanded.flatten(),
            torch.tensor([10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0]),
        )

    def test_one_repeat_returns_the_original_tensor(self) -> None:
        kv = torch.randn(2, 8, 5, 8)
        self.assertIs(repeat_kv(kv, repeats=1), kv)

    def test_gqa_projection_widths(self) -> None:
        attention = CausalSelfAttention(attention_config(n_kv_heads=2))
        self.assertEqual(attention.q_proj.out_features, 64)
        self.assertEqual(attention.k_proj.out_features, 16)
        self.assertEqual(attention.v_proj.out_features, 16)
        self.assertEqual(attention.out_proj.in_features, 64)

    def test_parameter_reduction_matches_formula(self) -> None:
        mha = CausalSelfAttention(attention_config(n_kv_heads=8))
        gqa = CausalSelfAttention(attention_config(n_kv_heads=2))
        mha_parameters = sum(parameter.numel() for parameter in mha.parameters())
        gqa_parameters = sum(parameter.numel() for parameter in gqa.parameters())
        expected_reduction = 2 * 64 * (64 - 16)
        self.assertEqual(mha_parameters - gqa_parameters, expected_reduction)

    def test_gqa_forward_backward_shape_and_gradients(self) -> None:
        attention = CausalSelfAttention(attention_config(n_kv_heads=2))
        x = torch.randn(3, 11, 64, requires_grad=True)
        output = attention(x)
        self.assertEqual(output.shape, x.shape)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in attention.parameters()
            )
        )

    def test_small_overfit_model_preserves_head_sharing_ratio(self) -> None:
        self.assertEqual(small_attention_heads(8, 8), (4, 4))
        self.assertEqual(small_attention_heads(8, 2), (4, 1))
        with self.assertRaisesRegex(ValueError, "sharing ratio"):
            small_attention_heads(12, 2)

    def test_token_by_token_cache_matches_full_causal_attention(self) -> None:
        torch.manual_seed(7)
        attention = CausalSelfAttention(attention_config(n_kv_heads=2)).eval()
        x = torch.randn(2, 9, 64)
        cache = LayerKVCache(
            2, 2, 9, 8, device=x.device, dtype=x.dtype
        )
        with torch.no_grad():
            full = attention(x)
            cached = torch.cat(
                [attention(x[:, position : position + 1], cache) for position in range(9)],
                dim=1,
            )
        torch.testing.assert_close(cached, full, atol=1e-5, rtol=1e-5)
        self.assertEqual(cache.length, 9)
        self.assertEqual(cache.keys.shape[1], 2)

    def test_chunked_cache_matches_full_causal_attention(self) -> None:
        torch.manual_seed(11)
        attention = CausalSelfAttention(attention_config(n_kv_heads=2)).eval()
        x = torch.randn(1, 10, 64)
        cache = LayerKVCache(
            1, 2, 10, 8, device=x.device, dtype=x.dtype
        )
        with torch.no_grad():
            full = attention(x)
            cached = torch.cat(
                (
                    attention(x[:, :4], cache),
                    attention(x[:, 4:7], cache),
                    attention(x[:, 7:], cache),
                ),
                dim=1,
            )
        torch.testing.assert_close(cached, full, atol=1e-5, rtol=1e-5)

    def test_invalid_repeat_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            repeat_kv(torch.zeros(2, 3, 4), repeats=2)
        with self.assertRaisesRegex(ValueError, "positive"):
            repeat_kv(torch.zeros(1, 2, 3, 4), repeats=0)


if __name__ == "__main__":
    unittest.main()
