import tempfile
import unittest
from pathlib import Path

import torch

from llm.checkpoint import load_checkpoint, save_checkpoint
from llm.config import ExperimentConfig, ModelConfig, TrainingConfig
from llm.data import TokenCorpus, prepare_char_corpus
from llm.model import GPT
from llm.training import (
    create_grad_scaler,
    create_optimizer,
    learning_rate_at,
    train_micro_batches,
)


def small_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=16,
        context_length=8,
        n_layers=1,
        d_model=16,
        n_heads=2,
        n_kv_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
    )


def small_training_config() -> TrainingConfig:
    return TrainingConfig(
        device="cpu",
        precision="fp32",
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=10,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_steps=2,
        eval_interval=2,
        eval_batches=2,
        log_interval=1,
        checkpoint_interval=2,
    )


class TrainingTests(unittest.TestCase):
    def test_cosine_schedule_boundaries(self) -> None:
        config = small_training_config()
        self.assertAlmostEqual(learning_rate_at(0, config), 5e-4)
        self.assertAlmostEqual(learning_rate_at(1, config), 1e-3)
        self.assertAlmostEqual(learning_rate_at(2, config), 1e-3)
        self.assertAlmostEqual(learning_rate_at(10, config), 1e-4)

    def test_optimizer_groups_cover_parameters_once(self) -> None:
        model = GPT(small_model_config())
        optimizer = create_optimizer(model, small_training_config())
        grouped = [p for group in optimizer.param_groups for p in group["params"]]
        expected = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual({id(p) for p in grouped}, {id(p) for p in expected})
        self.assertEqual(len(grouped), len(expected))
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)

    def test_one_optimizer_update_changes_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "text.txt"
            source.write_text("abcdefghijklmnop" * 20, encoding="utf-8")
            prepare_char_corpus(source, root / "processed", 0.8)
            corpus = TokenCorpus(root / "processed")

            model = GPT(small_model_config())
            training = small_training_config()
            optimizer = create_optimizer(model, training)
            scaler = create_grad_scaler(torch.device("cpu"), training.precision)
            before = model.token_embedding.weight.detach().clone()
            loss, grad_norm = train_micro_batches(
                model,
                corpus,
                optimizer,
                scaler,
                training,
                context_length=8,
                device=torch.device("cpu"),
            )
            self.assertTrue(torch.isfinite(torch.tensor(loss)))
            self.assertGreater(grad_norm, 0)
            self.assertFalse(torch.equal(before, model.token_embedding.weight))

    def test_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            model = GPT(small_model_config())
            training = small_training_config()
            experiment = ExperimentConfig(model=small_model_config(), training=training)
            optimizer = create_optimizer(model, training)
            scaler = create_grad_scaler(torch.device("cpu"), training.precision)
            expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

            save_checkpoint(path, model, optimizer, scaler, experiment, 7, 2.5)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(1)
            checkpoint = load_checkpoint(path, model, optimizer, scaler)

            self.assertEqual(checkpoint["step"], 7)
            self.assertEqual(checkpoint["best_val_loss"], 2.5)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[name])
            self.assertFalse(path.with_suffix(".pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
