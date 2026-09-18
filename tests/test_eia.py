import tempfile
import unittest
from pathlib import Path

from llm.eia import (
    connect,
    insert_observations,
    observation_from_record,
    percent_change,
    series_values,
    year_over_year,
)


class EIATests(unittest.TestCase):
    def test_normalizes_eia_record_and_skips_missing_values(self) -> None:
        record = {
            "period": "2025-01",
            "series": "MCRFPP21",
            "value": "123.4",
            "unit": "Thousand Barrels",
            "area-name": "PADD 2",
            "product-name": "Crude Oil",
            "series-description": "Field production",
        }
        observation = observation_from_record("crude-production", record, "https://eia.gov")
        assert observation is not None
        self.assertEqual(observation.series_id, "MCRFPP21")
        self.assertEqual(observation.value, 123.4)
        self.assertIsNone(
            observation_from_record("x", {"period": "2025-01", "value": None}, "url")
        )

    def test_insert_upsert_and_query_series(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = connect(str(Path(directory) / "eia.sqlite"))
            records = [
                {"period": "2024-01", "series": "S1", "value": "10", "unit": "u"},
                {"period": "2025-01", "series": "S1", "value": "15", "unit": "u"},
            ]
            observations = [
                observation_from_record("x", record, "url") for record in records
            ]
            inserted = insert_observations(connection, [item for item in observations if item], "now")
            self.assertEqual(inserted, 2)
            self.assertEqual([row["period"] for row in series_values(connection, "S1")], ["2024-01", "2025-01"])

    def test_year_over_year_and_zero_percent_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = connect(str(Path(directory) / "eia.sqlite"))
            observations = [
                observation_from_record("x", {"period": "2024-01", "series": "S1", "value": "10"}, "url"),
                observation_from_record("x", {"period": "2025-01", "series": "S1", "value": "15"}, "url"),
            ]
            insert_observations(connection, [item for item in observations if item], "now")
            result = year_over_year(connection, "S1", "2025-01")
            assert result is not None
            self.assertEqual(result["change"], 5.0)
            self.assertEqual(result["percent_change"], 50.0)
            self.assertIsNone(percent_change(0.0, 2.0))


if __name__ == "__main__":
    unittest.main()
