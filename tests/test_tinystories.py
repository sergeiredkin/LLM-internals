import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from llm.data import TokenCorpus
from llm.tinystories import extract_documents, iter_jsonl_texts, tokenize_documents
from llm.tokenizer import BPETokenizer, load_tokenizer, tokenizer_from_json


class TinyStoriesTests(unittest.TestCase):
    def test_bpe_round_trip_and_document_tokens(self) -> None:
        texts = [
            "Once upon a time, a little cat played in the garden.",
            "The little dog was happy and kind.",
        ] * 20
        tokenizer = BPETokenizer.train(iter(texts), vocab_size=128)
        encoded = tokenizer.encode(texts[0])
        self.assertEqual(tokenizer.decode(encoded), texts[0])
        document = tokenizer.encode(texts[0], add_document_tokens=True)
        self.assertEqual(document[0], tokenizer.bos_id)
        self.assertEqual(document[-1], tokenizer.eos_id)
        restored = tokenizer_from_json(tokenizer.tokenizer.to_str())
        self.assertEqual(restored.decode(restored.encode(texts[0])), texts[0])

    def test_extract_tokenize_and_load_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "stories.parquet"
            pq.write_table(
                pa.table({"text": ["A small cat sat.", "", "A kind dog ran."]}), parquet
            )
            documents = root / "documents.jsonl"
            count = extract_documents(parquet, documents, limit=2)
            self.assertEqual(count, 2)
            texts = list(iter_jsonl_texts(documents))
            self.assertEqual(texts, ["A small cat sat.", "A kind dog ran."])

            tokenizer = BPETokenizer.train(iter(texts * 10), vocab_size=64)
            processed = root / "processed"
            tokenizer.save(processed / "tokenizer.json")
            train_docs, train_tokens = tokenize_documents(
                documents, processed / "train.bin", tokenizer
            )
            val_docs, val_tokens = tokenize_documents(
                documents, processed / "val.bin", tokenizer
            )
            metadata = {
                "format": "uint16",
                "vocab_size": tokenizer.vocab_size,
                "train_documents": train_docs,
                "val_documents": val_docs,
                "train_tokens": train_tokens,
                "val_tokens": val_tokens,
            }
            (processed / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            loaded = load_tokenizer(processed / "tokenizer.json")
            self.assertIsInstance(loaded, BPETokenizer)
            corpus = TokenCorpus(processed)
            inputs, targets = corpus.get_batch("train", 1, context_length=3)
            self.assertEqual(inputs.shape, (1, 3))
            self.assertTrue((inputs[:, 1:] == targets[:, :-1]).all())


if __name__ == "__main__":
    unittest.main()
