import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_petroleum_chunks import sample_chunks
from llm.rag import (
    BM25Retriever,
    Chunk,
    assemble_context,
    Document,
    chunk_document,
    load_jsonl_chunks,
    load_jsonl_documents,
    retrieval_metrics,
    expand_petroleum_query,
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

    def test_petroleum_query_expansion_is_conservative(self) -> None:
        expanded = expand_petroleum_query("How does salt act as a seal?")
        self.assertIn("salt", expanded)
        self.assertIn("evaporite", expanded)
        self.assertIn("role", expanded)
        self.assertIn("objective", expand_petroleum_query("What was the purpose?"))
        self.assertIn("probable", expand_petroleum_query("What is likely?"))

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

    def test_pre_chunked_jsonl_loads_without_rechunking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.jsonl"
            path.write_text(json.dumps({
                "chunk_id": "page#chunk-0000",
                "document_id": "page",
                "text": "one two three",
                "source": "report.pdf",
                "title": "Report",
                "metadata": {"page": 7},
            }) + "\n", encoding="utf-8")
            chunks = load_jsonl_chunks(path)
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].chunk_id, "page#chunk-0000")
            self.assertEqual(chunks[0].metadata["page"], "7")

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

    def test_grouped_retrieval_keeps_one_chunk_per_parent_document(self) -> None:
        chunks = [
            Chunk("page-a-1", "page-a", "pressure pressure drilling", "", "", {}),
            Chunk("page-a-2", "page-a", "pressure drilling", "", "", {}),
            Chunk("page-b-1", "page-b", "pressure drilling", "", "", {}),
        ]
        results = BM25Retriever(chunks).retrieve(
            "pressure drilling", top_k=2, group_by_document=True
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(len({item.chunk.document_id for item in results}), 2)

    def test_context_assembly_has_citations_and_source_provenance(self) -> None:
        chunks = [chunk_document(document, chunk_size=100, overlap=0)[0] for document in self.documents]
        result = assemble_context(
            BM25Retriever(chunks), "drilling pressure cuttings", top_k=1
        )
        self.assertFalse(result.abstained)
        self.assertEqual(result.citations, ("[1]",))
        self.assertIn("[1] Drilling fluids — manual.pdf", result.context)
        self.assertIn("cuttings", result.context)
        self.assertEqual(result.results[0].chunk.document_id, "drilling")

    def test_petroleum_context_includes_source_id_and_page(self) -> None:
        chunk = Chunk(
            "petroleum#page-7#chunk-0", "petroleum#page-7", "reservoir pressure", "report.pdf",
            "Petroleum report", {"source_id": "usgs-001", "page": "7"},
        )
        result = assemble_context(BM25Retriever([chunk]), "reservoir pressure")
        self.assertIn("[source_id: usgs-001]", result.context)
        self.assertIn("[page: 7]", result.context)

    def test_context_abstains_without_lexical_evidence(self) -> None:
        chunks = [chunk_document(document, chunk_size=100, overlap=0)[0] for document in self.documents]
        result = assemble_context(BM25Retriever(chunks), "quantum chromodynamics")
        self.assertTrue(result.abstained)
        self.assertEqual(result.context, "")
        self.assertIn("no supporting", result.reason or "")

    def test_context_character_budget_and_threshold(self) -> None:
        chunks = [chunk_document(document, chunk_size=100, overlap=0)[0] for document in self.documents]
        retriever = BM25Retriever(chunks)
        result = assemble_context(
            retriever,
            "pressure production",
            top_k=3,
            max_characters=80,
            minimum_score=0.1,
        )
        self.assertLessEqual(len(result.context), 80)
        with self.assertRaisesRegex(ValueError, "max_characters"):
            assemble_context(retriever, "pressure", max_characters=0)

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

    def test_review_sampling_spans_the_whole_corpus(self) -> None:
        chunks = [
            Chunk(str(index), str(index), f"text {index}", "", "", {})
            for index in range(10)
        ]
        selected = sample_chunks(chunks, 4)
        self.assertEqual([chunk.chunk_id for chunk in selected], ["0", "3", "6", "9"])

    def test_invalid_arguments_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_document(self.documents[0], chunk_size=4, overlap=4)
        with self.assertRaisesRegex(ValueError, "at least one"):
            BM25Retriever([])
        with self.assertRaisesRegex(ValueError, "top_k"):
            BM25Retriever([chunk_document(self.documents[0])[0]]).retrieve("x", top_k=0)


if __name__ == "__main__":
    unittest.main()
