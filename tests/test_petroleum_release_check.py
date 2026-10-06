import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PetroleumReleaseCheckTests(unittest.TestCase):
    def write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    def make_release(self, root: Path) -> Path:
        data = root / "petroleum"
        data.mkdir()
        manifest = data / "manifest.jsonl"
        self.write_jsonl(manifest, [
            {"source_id": "src-a", "url": "https://example.test/a.pdf", "license": "public-domain"},
            {"source_id": "src-b", "url": "https://example.test/b.pdf", "license": "public-domain"},
        ])
        sourced_records = [
            {"document_id": "src-a#page-0001", "metadata": {"source_id": "src-a"}},
            {"document_id": "src-b#page-0001", "metadata": {"source_id": "src-b"}},
        ]
        for name in ("pages-expanded.jsonl", "chunks-expanded-clean.jsonl"):
            self.write_jsonl(data / name, sourced_records)
        table_facts = [
            {"document_id": "src-a#page-0001", "metadata": {"source_id": "src-a"}},
            {"document_id": "src-b#page-0001", "metadata": {"source_id": "src-b"}},
        ]
        self.write_jsonl(data / "table-fact-candidates.jsonl", table_facts)
        self.write_jsonl(data / "chunks-augmented.jsonl", sourced_records + table_facts)
        self.write_jsonl(data / "queries-50.jsonl", [{"query": str(index)} for index in range(50)])
        self.write_jsonl(data / "gold-answers-50.jsonl", [{"answer": str(index)} for index in range(50)])
        return manifest

    def run_check(self, manifest: Path, data: Path) -> subprocess.CompletedProcess:
        return subprocess.run([
            sys.executable,
            "-m",
            "scripts.check_petroleum_release",
            "--manifest",
            str(manifest),
            "--data-dir",
            str(data),
            "--expected-sources",
            "2",
            "--minimum-table-facts",
            "2",
        ], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)

    def test_complete_release_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_release(Path(directory))
            result = self.run_check(manifest, manifest.parent)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "pass")

    def test_partial_source_artifacts_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_release(Path(directory))
            self.write_jsonl(manifest.parent / "pages-expanded.jsonl", [
                {"document_id": "src-a#page-0001", "metadata": {"source_id": "src-a"}},
            ])
            result = self.run_check(manifest, manifest.parent)
        payload = json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(payload["status"], "fail")
        self.assertTrue(any("pages-expanded.jsonl" in error for error in payload["errors"]))

    def test_missing_table_facts_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_release(Path(directory))
            self.write_jsonl(manifest.parent / "table-fact-candidates.jsonl", [])
            result = self.run_check(manifest, manifest.parent)
        payload = json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(payload["status"], "fail")
        self.assertTrue(any("table-fact-candidates.jsonl" in error for error in payload["errors"]))


if __name__ == "__main__":
    unittest.main()
