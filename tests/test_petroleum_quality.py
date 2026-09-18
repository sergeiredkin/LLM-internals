import unittest

from llm.petroleum_quality import quality_flags


class PetroleumQualityTests(unittest.TestCase):
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
