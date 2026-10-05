#!/usr/bin/env python3
"""Build verified gold answers for the 50-question petroleum benchmark.

Takes the pipeline-validation answer drafts (answer-eval-50.jsonl, all marked
review_required) and produces a reviewed gold set:

- For every draft answer, finds the best matching verbatim window in the cited
  evidence (word-level sliding match). Only windows with containment >= 0.75
  are kept, so every gold answer sentence is literally present in a cited
  source page: extractive grounding by construction.
- Drops page-header/title noise that leaked into the drafts, because the gold
  text is taken from the evidence, not from the draft.
- Re-cites only the evidence chunks that actually contributed sentences.
- Runs llm.answer_quality.evaluate_answer on each record; grounded records are
  marked status=verified, everything else status=review with reasons.

Output: data/petroleum/gold-answers-50.jsonl (+ markdown summary in reports/results/).
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from datetime import date
from pathlib import Path

from llm.answer_quality import evaluate_answer

DATA = Path("data/petroleum")
RESULTS = Path("reports/results")
INPUT = DATA / "answer-eval-50.jsonl"
OUTPUT = DATA / "gold-answers-50.jsonl"
REPORT = RESULTS / "petroleum-gold-answers-50.md"

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
MIN_RATIO = 0.75


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def normalize_word(word: str) -> str:
    """Lowercase a single word, keeping 1:1 alignment with the original text."""
    return re.sub(r"[^A-Za-z0-9'-]", "", word).lower()


def dehyphenate(text: str) -> str:
    """Rejoin words hyphenated across PDF line breaks: 'Forma- tions' -> 'Formations'.

    Only hyphens followed by whitespace and a lowercase word are joined, so
    genuine compounds ('oil-gas', 'source-rock') are untouched.
    """
    return re.sub(r"([A-Za-z])-\s+(?=[a-z])", r"\1", text)


def best_window(candidate: str, evidence_text: str) -> tuple[str, float]:
    """Find the evidence word-window most similar to the candidate sentence.

    Uses word-aligned difflib ratios over sliding windows so the returned text
    preserves the evidence's original casing, numbers, and units verbatim.
    """
    cand = tokens(dehyphenate(candidate))
    if not cand:
        return "", 0.0
    words = dehyphenate(evidence_text).split()
    if not words:
        return "", 0.0
    ev = [normalize_word(word) for word in words]
    width = len(cand)
    best_text, best_score = "", 0.0
    step = max(width // 2, 1)
    matcher = difflib.SequenceMatcher(None, cand, autojunk=False)
    for start in range(0, max(len(ev) - width + 1, 1), step):
        window = ev[start:start + width]
        matcher.set_seq2(window)
        score = matcher.ratio()
        if score > best_score:
            best_score = score
            best_text = " ".join(words[start:start + width])
    return best_text, best_score


def dedupe_overlaps(new_tokens: set[str], kept: list[str]) -> bool:
    for existing in kept:
        existing_tokens = set(tokens(existing))
        if new_tokens and existing_tokens:
            overlap = len(new_tokens & existing_tokens) / max(
                len(new_tokens | existing_tokens), 1
            )
            if overlap > 0.5:
                return False
    return True


HEADER_RE = re.compile(
    r"^(?:\d{1,3}\s+\S|Table\s+\d|Figure\s+\d|\d+\s*$|[IVXLC]+\s*$)", re.IGNORECASE
)


def build_gold_record(record: dict) -> dict:
    evidence = record.get("evidence") or []
    # De-hyphenate PDF line breaks, drop standalone header furniture lines
    # (page numbers, running titles, "Table N." captions), then join the rest
    # into one paragraph so sentence boundaries are recovered before matching.
    raw = dehyphenate(str(record.get("answer", "")))
    content_lines = [
        line.strip() for line in raw.splitlines()
        if line.strip() and not HEADER_RE.match(line.strip())
    ]
    paragraph = re.sub(r"\s+", " ", " ".join(content_lines)).strip()
    units = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]

    gold_parts: list[str] = []
    cited_ids: list[str] = []
    weak_sentences: list[str] = []
    for sentence in units:
        best = ("", "", 0.0)  # text, doc_id, score
        for item in evidence:
            window, score = best_window(sentence, str(item.get("text", "")))
            if score > best[2]:
                best = (window, str(item.get("document_id", "")), score)
        if best[2] >= MIN_RATIO and best[0]:
            if dedupe_overlaps(set(tokens(best[0])), gold_parts):
                gold_parts.append(best[0])
                if best[1] not in cited_ids:
                    cited_ids.append(best[1])
        else:
            weak_sentences.append(sentence)

    used_evidence = [
        {**item, "text": dehyphenate(str(item.get("text", "")))}
        for item in evidence
        if str(item.get("document_id", "")) in cited_ids
    ]
    gold_answer = " ".join(gold_parts)
    expected_abstention = bool(record.get("expected_abstention", False))
    if expected_abstention and not gold_answer:
        # An expected abstention with no extractable content is correct by definition.
        return {
            "query": record.get("query", ""),
            "answer": "",
            "citations": [],
            "evidence": [],
            "expected_abstention": True,
            "status": "verified",
            "review_reasons": [],
            "checks": {"abstention_correct": True, "grounded": True},
            "source_draft": True,
        }
    checks = evaluate_answer(
        {
            "query": record.get("query", ""),
            "answer": gold_answer,
            "citations": cited_ids,
            "evidence": used_evidence,
            "expected_abstention": expected_abstention,
        }
    )
    reasons = []
    if weak_sentences:
        reasons.append(f"{len(weak_sentences)} unsupported draft sentence(s)")
    if not gold_answer:
        reasons.append("no supported content found in cited evidence")
    if not checks["grounded"]:
        reasons.append(
            f"quality checks: support={checks['lexical_support']:.2f}, "
            f"numeric={checks['numeric_grounding']:.2f}, "
            f"abstention_correct={checks['abstention_correct']}"
        )
    status = "verified" if gold_answer and not reasons else "review"
    return {
        "query": record.get("query", ""),
        "answer": gold_answer,
        "citations": cited_ids,
        "evidence": used_evidence,
        "expected_abstention": bool(record.get("expected_abstention", False)),
        "status": status,
        "review_reasons": reasons,
        "checks": checks,
        "source_draft": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    gold = [build_gold_record(record) for record in records]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in gold:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    verified = sum(1 for r in gold if r["status"] == "verified")
    abstention = sum(1 for r in gold if r["expected_abstention"])
    numeric_fail = sum(1 for r in gold if r["status"] == "review" and r["checks"]["numeric_grounding"] < 1.0)
    support_fail = sum(1 for r in gold if r["status"] == "review" and r["checks"]["lexical_support"] < 0.2)
    report = f"""# Gold Answers: 50-Question Petroleum Benchmark

Built {date.today().isoformat()} by `scripts/build_gold_answers.py` from the pipeline drafts in
`data/petroleum/answer-eval-50.jsonl`. Every gold answer is extractive: each sentence is a
verbatim window (similarity >= {MIN_RATIO:.2f}) from a cited evidence page, so grounding
holds by construction rather than by trust.

## Status

```text
Verified:    {verified}/{len(gold)}
Needs review: {len(gold) - verified}
Abstention:   {abstention}
Review causes: numeric grounding {numeric_fail}, lexical support {support_fail}
```

Records with `status: review` list machine-readable reasons in `review_reasons`
and still require human sign-off before use as gold labels.
"""
    args.report.write_text(report, encoding="utf-8")
    print(f"gold answers: {args.output}")
    print(f"verified: {verified}/{len(gold)}  (review: {len(gold) - verified})")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
