import csv
import tempfile
import unittest
from pathlib import Path

from llm.bsee import connect, insert, read_csv


class BSEETests(unittest.TestCase):
    def test_import_requires_explicit_schema_and_preserves_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bsee.csv"
            fields = ["source_id", "period", "area", "operator", "oil_bbl", "gas_mcf", "water_bbl", "source_url", "source_checksum"]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(source_id="bsee-pilot", period="2025-01", area="GOM",
                                     operator="Example", oil_bbl="10", gas_mcf="20",
                                     water_bbl="", source_url="https://bsee.gov/file.csv",
                                     source_checksum="abc"))
            rows = read_csv(path)
            self.assertEqual(rows[0].oil_bbl, 10.0)
            self.assertIsNone(rows[0].water_bbl)
            connection = connect(":memory:")
            self.assertEqual(insert(connection, rows), 1)
            saved = connection.execute("SELECT source_url, source_checksum FROM bsee_production").fetchone()
            self.assertEqual(tuple(saved), ("https://bsee.gov/file.csv", "abc"))

    def test_missing_column_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("period,oil_bbl\n2025-01,10\n")
            with self.assertRaises(ValueError):
                read_csv(path)


if __name__ == "__main__":
    unittest.main()
