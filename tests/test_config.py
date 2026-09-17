import tempfile
import unittest
from pathlib import Path

from llm.config import ExperimentConfig, ModelConfig, config_from_dict, load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_shakespeare_config_loads(self) -> None:
        config = load_config(ROOT / "configs" / "shakespeare.yaml")
        self.assertEqual(config.name, "shakespeare-char")
        self.assertEqual(config.model.head_dim, 64)
        self.assertEqual(config.training.effective_batch_size, 64)
        self.assertEqual(config.tokens_per_update, 16_384)

    def test_dictionary_round_trip(self) -> None:
        original = load_config(ROOT / "configs" / "shakespeare.yaml")
        restored = config_from_dict(original.to_dict())
        self.assertEqual(restored, original)

    def test_invalid_head_count_is_rejected(self) -> None:
        config = ExperimentConfig(model=ModelConfig(d_model=384, n_heads=5))
        with self.assertRaisesRegex(ValueError, "divisible"):
            config.validate()

    def test_rope_requires_even_head_dimension(self) -> None:
        config = ExperimentConfig(
            model=ModelConfig(
                d_model=30, n_heads=10, n_kv_heads=10, position_encoding="rope"
            )
        )
        with self.assertRaisesRegex(ValueError, "even"):
            config.validate()

    def test_invalid_mlp_type_is_rejected(self) -> None:
        config = ExperimentConfig(model=ModelConfig(mlp_type="unknown"))
        with self.assertRaisesRegex(ValueError, "mlp_type"):
            config.validate()

    def test_legacy_config_defaults_to_gelu(self) -> None:
        raw = load_config(ROOT / "configs" / "shakespeare.yaml").to_dict()
        raw["model"].pop("mlp_type", None)
        raw["model"].pop("swiglu_multiple_of", None)
        restored = config_from_dict(raw)
        self.assertEqual(restored.model.mlp_type, "gelu")
        self.assertEqual(restored.model.swiglu_multiple_of, 16)

    def test_unknown_top_level_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("name: test\nunknown: true\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown top-level"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
