"""Conservative quality checks for extracted petroleum report chunks."""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_TABLE_MARKERS = (
    "table",
    "play type",
    "reservoir",
    "hydrocarbon",
    "probable",
    "remarks",
    "areal extent",
)


def table_candidate_score(text: str) -> int:
    """Score pages that should be preserved for a separate table retrieval path."""

    lowered = text.lower()
    marker_count = sum(lowered.count(marker) for marker in _TABLE_MARKERS)
    uppercase_words = len(re.findall(r"\b[A-Z][A-Z-]{2,}\b", text))
    return marker_count + min(uppercase_words, 5)


def reconstruct_table_rows(text: str) -> list[str]:
    """Split common extracted table prose into auditable row-like records.

    PDF extraction rarely preserves columns. These row-like spans retain source wording while
    avoiding fabricated column alignment; callers should treat them as candidates for review.
    """

    normalized = re.sub(r"[ \t]+", " ", text.replace("\u00ad", "")).strip()
    if not normalized:
        return []
    markers = r"(?=(?:If present|Possible play|Summary of|Table \d+\.?))"
    rows = [part.strip(" \n") for part in re.split(markers, normalized, flags=re.IGNORECASE) if part.strip()]
    return rows or [normalized]


def quality_flags(text: str, *, minimum_characters: int = 200) -> list[str]:
    """Return rejection flags without pretending to judge scientific correctness."""

    stripped = text.strip()
    if not stripped:
        return ["empty"]
    words = _WORD_RE.findall(stripped)
    flags: list[str] = []
    if len(stripped) < minimum_characters or len(words) < 35:
        flags.append("too_short")

    beginning = stripped[:240].lower()
    figure_words = sum(
        beginning.count(term) for term in ("figure", "fig.", "map", "plate", "cross section")
    )
    if figure_words and len(words) < 70:
        flags.append("figure_or_map_only")

    if re.search(r"^(references|bibliography)\b", beginning) and len(words) < 120:
        flags.append("bibliography")

    replacement_count = stripped.count("�")
    control_count = sum(ord(char) < 32 and char not in "\n\t\r" for char in stripped)
    if replacement_count or control_count or (len(stripped) > 100 and sum(not char.isprintable() for char in stripped) > 3):
        flags.append("corrupted_text")

    # Many non-word symbols in a short span are typical of an extracted map/image rather than prose.
    if len(stripped) >= 100 and len(words) < 25:
        symbol_count = sum(not char.isalnum() and not char.isspace() for char in stripped)
        if symbol_count / len(stripped) > 0.25:
            flags.append("sparse_symbols")
    return list(dict.fromkeys(flags))
