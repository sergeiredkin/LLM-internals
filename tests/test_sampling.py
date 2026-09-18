import unittest

import torch

from llm.sampling import sample_next_token, top_k_filter, top_p_filter


class SamplingTests(unittest.TestCase):
    def test_top_k_keeps_exactly_k_logits(self) -> None:
        logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
        filtered = top_k_filter(logits, 2)
        self.assertEqual(torch.isfinite(filtered).sum().item(), 2)
        torch.testing.assert_close(filtered[0, [1, 3]], logits[0, [1, 3]])

    def test_top_k_larger_than_vocabulary_keeps_everything(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        torch.testing.assert_close(top_k_filter(logits, 10), logits)

    def test_top_p_keeps_smallest_set_reaching_threshold(self) -> None:
        probabilities = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
        filtered = top_p_filter(probabilities.log(), 0.6)
        expected_finite = torch.tensor([[True, True, False, False]])
        torch.testing.assert_close(torch.isfinite(filtered), expected_finite)

    def test_top_p_does_not_keep_an_extra_token_at_exact_threshold(self) -> None:
        probabilities = torch.tensor([[0.5, 0.3, 0.2]])
        filtered = top_p_filter(probabilities.log(), 0.5)
        torch.testing.assert_close(
            torch.isfinite(filtered), torch.tensor([[True, False, False]])
        )

    def test_top_p_restores_original_token_order_for_batches(self) -> None:
        probabilities = torch.tensor(
            [[0.1, 0.6, 0.2, 0.1], [0.45, 0.05, 0.10, 0.40]]
        )
        filtered = top_p_filter(probabilities.log(), 0.7)
        expected_finite = torch.tensor(
            [[False, True, True, False], [True, False, False, True]]
        )
        torch.testing.assert_close(torch.isfinite(filtered), expected_finite)

    def test_top_p_always_keeps_the_most_likely_token(self) -> None:
        logits = torch.tensor([[-2.0, 5.0, 1.0]])
        filtered = top_p_filter(logits, 0.001)
        self.assertEqual(torch.isfinite(filtered).sum().item(), 1)
        self.assertTrue(torch.isfinite(filtered[0, 1]))

    def test_top_p_one_disables_filtering(self) -> None:
        logits = torch.randn(2, 5)
        filtered = top_p_filter(logits, 1.0)
        torch.testing.assert_close(filtered, logits)
        self.assertIsNot(filtered, logits)

    def test_sampling_is_reproducible_with_a_generator(self) -> None:
        logits = torch.tensor([[2.0, 1.5, 1.0, 0.5]])
        first_generator = torch.Generator().manual_seed(17)
        second_generator = torch.Generator().manual_seed(17)
        first = sample_next_token(
            logits, temperature=0.8, top_k=3, top_p=0.8, generator=first_generator
        )
        second = sample_next_token(
            logits, temperature=0.8, top_k=3, top_p=0.8, generator=second_generator
        )
        torch.testing.assert_close(first, second)

    def test_top_k_one_is_deterministic_argmax(self) -> None:
        logits = torch.tensor([[1.0, 4.0, 2.0], [3.0, 2.0, 1.0]])
        sampled = sample_next_token(logits, top_k=1, top_p=0.1)
        torch.testing.assert_close(sampled, torch.tensor([[1], [0]]))

    def test_invalid_arguments_are_rejected(self) -> None:
        logits = torch.randn(1, 4)
        for top_p in (0.0, -0.1, 1.1):
            with self.subTest(top_p=top_p):
                with self.assertRaisesRegex(ValueError, "top_p"):
                    top_p_filter(logits, top_p)
        with self.assertRaisesRegex(ValueError, "top_k"):
            top_k_filter(logits, 0)
        with self.assertRaisesRegex(ValueError, "temperature"):
            sample_next_token(logits, temperature=0.0)
        with self.assertRaisesRegex(ValueError, "shape"):
            top_p_filter(torch.randn(4), 0.9)


if __name__ == "__main__":
    unittest.main()
