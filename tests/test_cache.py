from __future__ import annotations

import unittest

import torch

from llm.cache import LayerKVCache


class LayerKVCacheTests(unittest.TestCase):
    def make_cache(self, n_kv_heads: int = 2, max_length: int = 8) -> LayerKVCache:
        return LayerKVCache(
            batch_size=1,
            n_kv_heads=n_kv_heads,
            max_length=max_length,
            head_dim=4,
            device="cpu",
            dtype=torch.bfloat16,
        )

    def test_append_preserves_tokens_and_tracks_position(self) -> None:
        cache = self.make_cache()
        first_keys = torch.arange(24, dtype=torch.bfloat16).view(1, 2, 3, 4)
        first_values = first_keys + 100
        keys, values = cache.append(first_keys, first_values)
        self.assertEqual(cache.length, 3)
        self.assertEqual(cache.remaining, 5)
        torch.testing.assert_close(keys, first_keys)
        torch.testing.assert_close(values, first_values)

        second_keys = torch.full((1, 2, 2, 4), 7.0, dtype=torch.bfloat16)
        second_values = torch.full((1, 2, 2, 4), 9.0, dtype=torch.bfloat16)
        keys, values = cache.append(second_keys, second_values)
        self.assertEqual(cache.length, 5)
        torch.testing.assert_close(keys[:, :, :3], first_keys)
        torch.testing.assert_close(keys[:, :, 3:], second_keys)
        torch.testing.assert_close(values[:, :, :3], first_values)
        torch.testing.assert_close(values[:, :, 3:], second_values)

    def test_capacity_failure_does_not_change_length(self) -> None:
        cache = self.make_cache(max_length=3)
        cache.append(
            torch.zeros(1, 2, 2, 4, dtype=torch.bfloat16),
            torch.zeros(1, 2, 2, 4, dtype=torch.bfloat16),
        )
        with self.assertRaisesRegex(ValueError, "capacity exceeded"):
            cache.append(
                torch.zeros(1, 2, 2, 4, dtype=torch.bfloat16),
                torch.zeros(1, 2, 2, 4, dtype=torch.bfloat16),
            )
        self.assertEqual(cache.length, 2)

    def test_reset_reuses_allocated_storage(self) -> None:
        cache = self.make_cache()
        key_pointer = cache.keys.data_ptr()
        value_pointer = cache.values.data_ptr()
        cache.append(
            torch.randn(1, 2, 3, 4, dtype=torch.bfloat16),
            torch.randn(1, 2, 3, 4, dtype=torch.bfloat16),
        )
        cache.reset()
        self.assertEqual(cache.length, 0)
        self.assertEqual(cache.active()[0].shape[2], 0)
        self.assertEqual(cache.keys.data_ptr(), key_pointer)
        self.assertEqual(cache.values.data_ptr(), value_pointer)

    def test_gqa_cache_is_four_times_smaller_than_mha(self) -> None:
        gqa = self.make_cache(n_kv_heads=2)
        mha = self.make_cache(n_kv_heads=8)
        self.assertEqual(mha.memory_bytes, 4 * gqa.memory_bytes)

    def test_memory_accounting(self) -> None:
        cache = self.make_cache(n_kv_heads=2, max_length=8)
        expected = 2 * 1 * 2 * 8 * 4 * 2  # K/V * B * Hkv * T * D * BF16 bytes
        self.assertEqual(cache.memory_bytes, expected)

    def test_invalid_append_shape_and_dtype_are_rejected(self) -> None:
        cache = self.make_cache()
        valid = torch.zeros(1, 2, 1, 4, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "identical"):
            cache.append(valid, torch.zeros(1, 2, 2, 4, dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "expected shape"):
            cache.append(
                torch.zeros(1, 1, 1, 4, dtype=torch.bfloat16),
                torch.zeros(1, 1, 1, 4, dtype=torch.bfloat16),
            )
        with self.assertRaisesRegex(ValueError, "same dtype"):
            cache.append(valid.float(), valid.float())

    def test_invalid_constructor_arguments_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            LayerKVCache(1, 2, 0, 4, device="cpu", dtype=torch.float32)
        with self.assertRaisesRegex(TypeError, "floating point"):
            LayerKVCache(1, 2, 8, 4, device="cpu", dtype=torch.int64)


if __name__ == "__main__":
    unittest.main()
