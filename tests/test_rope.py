from __future__ import annotations

import unittest

import torch

from llm.rope import RotaryEmbedding, rotate_half


class RotaryEmbeddingTests(unittest.TestCase):
    def test_rotate_half_is_a_ninety_degree_rotation(self) -> None:
        vector = torch.tensor([3.0, 4.0])
        torch.testing.assert_close(rotate_half(vector), torch.tensor([-4.0, 3.0]))

    def test_position_zero_is_unchanged(self) -> None:
        rope = RotaryEmbedding(head_dim=4)
        q = torch.randn(2, 3, 1, 4)
        k = torch.randn(2, 1, 1, 4)
        rotated_q, rotated_k = rope(q, k)
        torch.testing.assert_close(rotated_q, q)
        torch.testing.assert_close(rotated_k, k)

    def test_rotation_preserves_norms(self) -> None:
        rope = RotaryEmbedding(head_dim=8)
        q = torch.randn(2, 4, 7, 8)
        k = torch.randn(2, 2, 7, 8)
        rotated_q, rotated_k = rope(q, k, position_offset=11)
        torch.testing.assert_close(
            torch.linalg.vector_norm(rotated_q, dim=-1),
            torch.linalg.vector_norm(q, dim=-1),
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(rotated_k, dim=-1),
            torch.linalg.vector_norm(k, dim=-1),
        )

    def test_dot_product_depends_on_relative_position(self) -> None:
        rope = RotaryEmbedding(head_dim=8)
        q = torch.randn(1, 1, 1, 8)
        k = torch.randn(1, 1, 1, 8)

        q_at_7, _ = rope(q, q, position_offset=7)
        k_at_10, _ = rope(k, k, position_offset=10)
        q_at_107, _ = rope(q, q, position_offset=107)
        k_at_110, _ = rope(k, k, position_offset=110)

        first_score = (q_at_7 * k_at_10).sum()
        shifted_score = (q_at_107 * k_at_110).sum()
        torch.testing.assert_close(first_score, shifted_score, atol=1e-5, rtol=1e-5)

    def test_different_q_and_k_head_counts_are_supported(self) -> None:
        rope = RotaryEmbedding(head_dim=8)
        q = torch.randn(2, 4, 6, 8)
        k = torch.randn(2, 2, 6, 8)
        rotated_q, rotated_k = rope(q, k)
        self.assertEqual(rotated_q.shape, q.shape)
        self.assertEqual(rotated_k.shape, k.shape)

    def test_bfloat16_dtype_is_preserved(self) -> None:
        rope = RotaryEmbedding(head_dim=8).to(dtype=torch.bfloat16)
        q = torch.randn(1, 4, 6, 8, dtype=torch.bfloat16)
        k = torch.randn(1, 2, 6, 8, dtype=torch.bfloat16)
        rotated_q, rotated_k = rope(q, k)
        self.assertEqual(rotated_q.dtype, torch.bfloat16)
        self.assertEqual(rotated_k.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(rotated_q).all())
        self.assertTrue(torch.isfinite(rotated_k).all())

    def test_gradients_are_finite(self) -> None:
        rope = RotaryEmbedding(head_dim=8)
        q = torch.randn(2, 4, 6, 8, requires_grad=True)
        k = torch.randn(2, 2, 6, 8, requires_grad=True)
        rotated_q, rotated_k = rope(q, k)
        (rotated_q.square().mean() + rotated_k.square().mean()).backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue(torch.isfinite(k.grad).all())

    def test_invalid_dimensions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive even"):
            RotaryEmbedding(head_dim=7)
        with self.assertRaisesRegex(ValueError, "even final dimension"):
            rotate_half(torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
