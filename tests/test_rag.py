import tempfile
import unittest
from pathlib import Path

from llm.rag import (
    BM25Retriever,
    Chunk,
    Document,
    chunk_document,
    load_jsonl_documents,
    retrieval_metrics,
    tokenize,
    write_jsonl_documents,
)


class RAGTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            Document(
                "drilling",
                "Drilling mud controls pressure and carries rock cuttings to the surface.",
                source="manual.pdf",
                title="Drilling fluids",
                metadata={"license": "CC-BY"},
            ),
            Document(
                "reservoir",
                "Reservoir pressure and permeability influence oil production and recovery.",
                source="reservoir.pdf",
                title="Reservoir engineering",
            ),
            Document(
                "safety",
                "A permit to work controls hazards during maintenance operations.",
                source="safety.pdf",
                title="Process safety",
            ),
        ]

    def test_tokenize_is_case_insensitive_and_stable(self) -> None:
        self.assertEqual(tokenize("API's Well-Log 12"), ["api's", "well-log", "12"])

    def test_chunking_preserves_provenance_and_overlap(self) -> None:
        document = Document("doc-1", "one two three four five six seven")
        chunks = chunk_document(document, chunk_size=4, overlap=1)
        self.assertEqual([chunk.text for chunk in chunks], [
            "one two three four",
            "four five six seven",
        ])
        self.assertEqual(chunks[0].chunk_id, "doc-1#chunk-0000")
        self.assertEqual(chunks[1].document_id, "doc-1")

    def test_jsonl_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "documents.jsonl"
            write_jsonl_documents(path, self.documents)
            loaded = load_jsonl_documents(path)
            self.assertEqual(loaded, self.documents)

    def test_bm25_retrieves_relevant_petroleum_document(self) -> None:
        chunks = [chunk_document(document, chunk_size=100, overlap=0)[0] for document in self.documents]
        retriever = BM25Retriever(chunks)
        result = retriever.retrieve("pressure drilling cuttings", top_k=1)
        self.assertEqual(result[0].chunk.document_id, "drilling")
        self.assertGreater(result[0].score, 0)

    def test_retrieval_tie_breaking_is_deterministic(self) -> None:
        chunks = [
            Chunk("b", "b", "same", "", "", {}),
            Chunk("a", "a", "same", "", "", {}),
        ]
        result = BM25Retriever(chunks).retrieve("absent", top_k=2)
        self.assertEqual([item.chunk.chunk_id for item in result], ["a", "b"])

    def test_retrieval_metrics(self) -> None:
        chunks = [chunk_document(document, chunk_size=100, overlap=0)[0] for document in self.documents]
        retriever = BM25Retriever(chunks)
        metrics = retrieval_metrics(
            retriever,
            [("drilling cuttings", {"drilling"}), ("permit hazards", {"safety"})],
            top_k=2,
        )
        self.assertEqual(metrics["hit_rate_at_k"], 1.0)
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(metrics["mrr_at_k"], 1.0)

    def test_invalid_arguments_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_document(self.documents[0], chunk_size=4, overlap=4)
        with self.assertRaisesRegex(ValueError, "at least one"):
            BM25Retriever([])
        with self.assertRaisesRegex(ValueError, "top_k"):
            BM25Retriever([chunk_document(self.documents[0])[0]]).retrieve("x", top_k=0)


if __name__ == "__main__":
    unittest.main()
