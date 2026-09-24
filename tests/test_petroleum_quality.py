import unittest

from llm.petroleum_quality import quality_flags, reconstruct_table_rows, table_candidate_score
from scripts.extract_petroleum_tables import merge_table_continuations


class PetroleumQualityTests(unittest.TestCase):
    def test_table_candidate_score_detects_petroleum_table_page(self) -> None:
        score = table_candidate_score(
            "Table 4. Petroleum geology and play analysis. PROBABLE MAJOR TYPE "
            "OF HYDROCARBON RESERVOIR THICKNESS REMARKS"
        )
        self.assertGreaterEqual(score, 4)

    def test_table_row_reconstruction_preserves_row_markers(self) -> None:
        rows = reconstruct_table_rows(
            "Table 4. Header If present in NPR4, dry gas. Summary of play."
        )
        self.assertEqual(len(rows), 3)
        self.assertIn("dry gas", rows[1])

    def test_table_continuations_merge_and_preserve_start_page(self) -> None:
        records = [
            {"document_id": "src#page-0055", "text": "Table 4. Petroleum geology", "metadata": {"source_id": "src", "page": "55"}},
            {"document_id": "src#page-0056", "text": "continued rows", "metadata": {"source_id": "src", "page": "56"}},
            {"document_id": "src#page-0057", "text": "Table 4. con't. more rows", "metadata": {"source_id": "src", "page": "57"}},
        ]
        merged = merge_table_continuations(records)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["document_id"], "src#page-0055")
        self.assertEqual(merged[0]["table_page_range"], ["55", "56", "57"])

    def test_good_prose_is_kept(self) -> None:
        text = " ".join(
            [
                "The reservoir consists of porous sandstone with effective porosity between",
                "eight and twelve percent. Permeability varies across the field, while shale",
                "acts as a regional seal. Hydrocarbon migration followed structural deformation",
                "and accumulated in closures identified by seismic interpretation." * 2,
            ]
        )
        self.assertEqual(quality_flags(text), [])

    def test_short_and_figure_only_chunks_are_rejected(self) -> None:
        flags = quality_flags("Figure 15. Map showing basin structure and wells.")
        self.assertIn("too_short", flags)
        self.assertIn("figure_or_map_only", flags)

    def test_bibliography_is_rejected_but_long_prose_reference_is_not(self) -> None:
        self.assertIn("bibliography", quality_flags("References\nSmith, 1984. Petroleum geology."))
        text = "The report references earlier work on reservoir quality. " + "Detailed analysis " * 40
        self.assertNotIn("bibliography", quality_flags(text))

    def test_corrupted_text_is_rejected(self) -> None:
        self.assertIn("corrupted_text", quality_flags("A" * 250 + " � "))

    def test_tables_are_not_rejected_just_for_symbols(self) -> None:
        text = "Table 2: Porosity (%) Permeability (md)\n" + "10 | 40\n" * 50
        flags = quality_flags(text)
        self.assertNotIn("figure_or_map_only", flags)
        self.assertNotIn("sparse_symbols", flags)


if __name__ == "__main__":
    unittest.main()
