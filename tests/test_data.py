import tempfile
import unittest
from pathlib import Path

import torch

from llm.data import TokenCorpus, prepare_char_corpus
from llm.tokenizer import CharTokenizer


class CharTokenizerTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        text = "to be\nor not to be"
        tokenizer = CharTokenizer.from_text(text)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_unknown_character_is_rejected(self) -> None:
        tokenizer = CharTokenizer.from_text("abc")
        with self.assertRaisesRegex(ValueError, "not in the vocabulary"):
            tokenizer.encode("abcd")


class TokenCorpusTests(unittest.TestCase):
    def test_prepare_load_and_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.txt"
            source.write_text("abcdefghijklmnopqrstuvwxyz" * 10, encoding="utf-8")
            output = root / "processed"

            metadata = prepare_char_corpus(source, output, train_fraction=0.8)
            corpus = TokenCorpus(output)
            self.assertEqual(metadata["vocab_size"], 26)
            self.assertEqual(len(corpus.train) + len(corpus.val), 260)

            generator = torch.Generator().manual_seed(123)
            inputs, targets = corpus.get_batch(
                "train", batch_size=4, context_length=12, generator=generator
            )
            self.assertEqual(inputs.shape, (4, 12))
            self.assertEqual(inputs.dtype, torch.long)
            torch.testing.assert_close(inputs[:, 1:], targets[:, :-1])

    def test_too_long_context_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.txt"
            source.write_text("abcdefghij", encoding="utf-8")
            output = root / "processed"
            prepare_char_corpus(source, output, train_fraction=0.5)
            corpus = TokenCorpus(output)
            with self.assertRaisesRegex(ValueError, "fewer than"):
                corpus.get_batch("val", batch_size=1, context_length=5)


if __name__ == "__main__":
    unittest.main()
