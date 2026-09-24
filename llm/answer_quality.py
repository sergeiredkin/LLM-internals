"""Answer-level checks for citation and grounding evaluation."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|feet|ft|m|km|Ma|MMBOE|bbl)?\b", re.IGNORECASE)


def answer_tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def evaluate_answer(record: dict) -> dict[str, object]:
    """Evaluate one answer against its supplied evidence records.

    Evidence is intentionally supplied by the caller so this checker never treats retrieval
    results as proof unless they were actually cited by the answer.
    """

    answer = str(record.get("answer", "")).strip()
    evidence = record.get("evidence") or []
    evidence_ids = {str(item.get("document_id", "")) for item in evidence}
    cited_ids = {str(item) for item in (record.get("citations") or [])}
    cited_ids.discard("")
    evidence_text = " ".join(str(item.get("text", "")) for item in evidence)
    answer_word_set = answer_tokens(answer)
    evidence_word_set = answer_tokens(evidence_text)
    overlap = len(answer_word_set & evidence_word_set) / max(len(answer_word_set), 1)
    numbers = [match.group(0).lower() for match in _NUMBER_RE.finditer(answer)]
    grounded_numbers = [number for number in numbers if number in evidence_text.lower()]
    abstention_expected = bool(record.get("expected_abstention", False))
    abstained = not answer
    return {
        "query": str(record.get("query", "")),
        "citation_coverage": (
            len(cited_ids & evidence_ids) / len(cited_ids) if cited_ids else 1.0
        ),
        "citation_support": 1.0 if cited_ids and cited_ids <= evidence_ids else 0.0,
        "lexical_support": overlap,
        "numeric_grounding": (
            len(grounded_numbers) / len(numbers) if numbers else 1.0
        ),
        "abstention_correct": abstained == abstention_expected,
        "grounded": bool(cited_ids <= evidence_ids and overlap >= 0.2 and
                          len(grounded_numbers) == len(numbers)),
    }
