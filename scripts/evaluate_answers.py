#!/usr/bin/env python3
"""Evaluate answer JSONL records for citation and grounding signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.answer_quality import evaluate_answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [evaluate_answer(json.loads(line)) for line in args.answers.open(encoding="utf-8") if line.strip()]
    if not results:
        raise SystemExit("answer file is empty")
    summary = {
        "answers": len(results),
        "citation_coverage": sum(float(r["citation_coverage"]) for r in results) / len(results),
        "citation_support": sum(float(r["citation_support"]) for r in results) / len(results),
        "lexical_support": sum(float(r["lexical_support"]) for r in results) / len(results),
        "sentence_support": sum(float(r["sentence_support"]) for r in results) / len(results),
        "unsupported_sentence_rate": sum(float(r["unsupported_sentence_rate"]) for r in results) / len(results),
        "numeric_grounding": sum(float(r["numeric_grounding"]) for r in results) / len(results),
        "abstention_accuracy": sum(bool(r["abstention_correct"]) for r in results) / len(results),
        "grounded_rate": sum(bool(r["grounded"]) for r in results) / len(results),
    }
    payload = {"summary": summary, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Result: {args.output}")


if __name__ == "__main__":
    main()
