import math
import unittest

import torch

from llm.config import ModelConfig
from llm.model import GPT


def tiny_config(**overrides) -> ModelConfig:
    values = {
        "vocab_size": 32,
        "context_length": 16,
        "n_layers": 2,
        "d_model": 32,
        "n_heads": 4,
        "n_kv_heads": 2,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
    }
    values.update(overrides)
    return ModelConfig(**values)


class GPTTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_forward_shape_and_loss(self) -> None:
        model = GPT(tiny_config())
        tokens = torch.randint(0, 32, (3, 11))
        targets = torch.randint(0, 32, (3, 11))
        logits, loss = model(tokens, targets)
        self.assertEqual(logits.shape, (3, 11, 32))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(loss.item(), math.log(32), delta=0.4)

    def test_future_tokens_do_not_change_past_logits(self) -> None:
        model = GPT(tiny_config()).eval()
        first = torch.randint(0, 32, (1, 12))
        second = first.clone()
        second[:, 7:] = torch.randint(0, 32, second[:, 7:].shape)
        logits_a, _ = model(first)
        logits_b, _ = model(second)
        torch.testing.assert_close(logits_a[:, :7], logits_b[:, :7])

    def test_embeddings_are_tied(self) -> None:
        model = GPT(tiny_config(tie_embeddings=True))
        self.assertEqual(
            model.token_embedding.weight.data_ptr(), model.lm_head.weight.data_ptr()
        )

    def test_backward_produces_finite_gradients(self) -> None:
        model = GPT(tiny_config())
        tokens = torch.randint(0, 32, (2, 8))
        _, loss = model(tokens, tokens)
        assert loss is not None
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_generate_has_requested_length(self) -> None:
        model = GPT(tiny_config()).eval()
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        output = model.generate(prompt, max_new_tokens=5, top_k=5)
        self.assertEqual(output.shape, (1, 8))
        torch.testing.assert_close(output[:, :3], prompt)

    def test_generation_stops_at_eos(self) -> None:
        model = GPT(tiny_config()).eval()
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        logits, _ = model(prompt)
        greedy_next_token = int(logits[0, -1].argmax())
        output = model.generate(
            prompt, max_new_tokens=5, top_k=1, eos_token_id=greedy_next_token
        )
        self.assertEqual(output.shape, (1, 4))
        self.assertEqual(output[0, -1].item(), greedy_next_token)

    def test_context_limit_is_enforced(self) -> None:
        model = GPT(tiny_config(context_length=4))
        with self.assertRaisesRegex(ValueError, "exceeds context length"):
            model(torch.zeros((1, 5), dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
