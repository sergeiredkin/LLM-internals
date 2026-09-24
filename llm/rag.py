"""Small dependency-free retrieval components for the petroleum RAG stage."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
_PETROLEUM_QUERY_EXPANSIONS = {
    "act": ("role", "function"),
    "hydrocarbons": ("oil", "gas", "petroleum"),
    "resource": ("oil-in-place", "deposit"),
    "effects": ("impact", "water-quality", "contamination"),
    "salt": ("seal", "evaporite", "anhydrite"),
    "aquifers": ("groundwater", "water-quality"),
}


def tokenize(text: str) -> list[str]:
    """Tokenize text consistently for indexing and queries."""

    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def expand_petroleum_query(query: str) -> str:
    """Add conservative petroleum vocabulary without changing original query terms."""

    terms = tokenize(query)
    additions = [term for token in terms for term in _PETROLEUM_QUERY_EXPANSIONS.get(token, ())]
    return query + (" " + " ".join(dict.fromkeys(additions)) if additions else "")


@dataclass(frozen=True)
class Document:
    """A source document before or after chunking."""

    document_id: str
    text: str
    source: str = ""
    title: str = ""
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class Chunk:
    """A retrievable text span with provenance back to its source document."""

    chunk_id: str
    document_id: str
    text: str
    source: str
    title: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class ContextResult:
    """Evidence assembled for a model prompt, or an explicit abstention."""

    query: str
    context: str
    citations: tuple[str, ...]
    results: tuple[RetrievalResult, ...]
    abstained: bool
    reason: str | None = None


def chunk_document(
    document: Document,
    *,
    chunk_size: int = 160,
    overlap: int = 32,
) -> list[Chunk]:
    """Split a document by token count while retaining source metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    words = document.text.split()
    if not words:
        return []
    step = chunk_size - overlap
    metadata = dict(document.metadata or {})
    chunks: list[Chunk] = []
    for index, start in enumerate(range(0, len(words), step)):
        text = " ".join(words[start : start + chunk_size])
        if not text:
            break
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}#chunk-{index:04d}",
                document_id=document.document_id,
                text=text,
                source=document.source,
                title=document.title,
                metadata=metadata,
            )
        )
        if start + chunk_size >= len(words):
            break
    return chunks


def load_jsonl_documents(path: str | Path) -> list[Document]:
    """Load documents from JSONL with required ``document_id`` and ``text`` fields."""

    documents: list[Document] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                documents.append(
                    Document(
                        document_id=str(record["document_id"]),
                        text=str(record["text"]),
                        source=str(record.get("source", "")),
                        title=str(record.get("title", "")),
                        metadata=(
                            None
                            if record.get("metadata") is None
                            else {
                                str(k): str(v)
                                for k, v in record["metadata"].items()
                            }
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid JSONL document at line {line_number}") from error
    return documents


def load_jsonl_chunks(path: str | Path) -> list[Chunk]:
    """Load an already chunked JSONL corpus without chunking it a second time."""

    chunks: list[Chunk] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                metadata = record.get("metadata") or {}
                chunks.append(
                    Chunk(
                        chunk_id=str(record["chunk_id"]),
                        document_id=str(record["document_id"]),
                        text=str(record["text"]),
                        source=str(record.get("source", "")),
                        title=str(record.get("title", "")),
                        metadata={str(k): str(v) for k, v in metadata.items()},
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid JSONL chunk at line {line_number}") from error
    return chunks


def write_jsonl_documents(path: str | Path, documents: list[Document]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")


class BM25Retriever:
    """Deterministic Okapi BM25 retriever over in-memory chunks."""

    def __init__(self, chunks: list[Chunk], *, k1: float = 1.2, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("at least one chunk is required")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be positive and b must be in [0, 1]")
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths)
        self.term_frequencies: list[dict[str, int]] = []
        document_frequency: dict[str, int] = {}
        for tokens in self.tokens:
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.term_frequencies.append(frequencies)
            for token in frequencies:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self.document_frequency = document_frequency
        self.document_count = len(chunks)

    def score(self, query: str, index: int) -> float:
        query_terms = set(tokenize(query))
        length = self.lengths[index]
        frequencies = self.term_frequencies[index]
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            df = self.document_frequency[term]
            idf = math.log(1.0 + (self.document_count - df + 0.5) / (df + 0.5))
            denominator = frequency + self.k1 * (
                1.0 - self.b + self.b * length / self.average_length
            )
            score += idf * frequency * (self.k1 + 1.0) / denominator
        return score

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        group_by_document: bool = False,
        expand_query: bool = False,
    ) -> list[RetrievalResult]:
        """Retrieve chunks, optionally keeping only the best chunk per parent document.

        Petroleum page records can produce several highly similar chunks. Grouping prevents
        one page from consuming the entire top-k and uses max-score aggregation for a stable
        parent-page ranking.
        """

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if expand_query:
            query = expand_petroleum_query(query)
        scored = [
            RetrievalResult(chunk, self.score(query, index))
            for index, chunk in enumerate(self.chunks)
        ]
        scored.sort(key=lambda result: (-result.score, result.chunk.chunk_id))
        if group_by_document:
            grouped: dict[str, RetrievalResult] = {}
            for result in scored:
                grouped.setdefault(result.chunk.document_id, result)
            scored = list(grouped.values())
        return scored[:top_k]


def assemble_context(
    retriever: BM25Retriever,
    query: str,
    *,
    top_k: int = 5,
    max_characters: int = 4000,
    minimum_score: float = 0.0,
    group_by_document: bool = True,
    expand_query: bool = False,
) -> ContextResult:
    """Create citation-labelled evidence, abstaining when lexical evidence is absent.

    Chunks are added in retrieval order until the character budget is reached. A query with no
    positive-scoring result abstains instead of emitting unrelated context.
    """

    if max_characters <= 0:
        raise ValueError("max_characters must be positive")
    if minimum_score < 0:
        raise ValueError("minimum_score cannot be negative")
    candidates = retriever.retrieve(
        query,
        top_k=top_k,
        group_by_document=group_by_document,
        expand_query=expand_query,
    )
    candidates = [result for result in candidates if result.score >= minimum_score]
    if not candidates or candidates[0].score <= 0.0:
        return ContextResult(
            query=query,
            context="",
            citations=(),
            results=(),
            abstained=True,
            reason="no supporting retrieval result",
        )

    sections: list[str] = []
    citations: list[str] = []
    selected: list[RetrievalResult] = []
    used = 0
    for result in candidates:
        citation = f"[{len(selected) + 1}]"
        header = f"{citation} {result.chunk.title or result.chunk.document_id}"
        if result.chunk.source:
            header += f" — {result.chunk.source}"
        page = (result.chunk.metadata or {}).get("page")
        source_id = (result.chunk.metadata or {}).get("source_id")
        if source_id:
            header += f" [source_id: {source_id}]"
        if page:
            header += f" [page: {page}]"
        section = f"{header}\n{result.chunk.text}"
        separator = "\n\n" if sections else ""
        if used + len(separator) + len(section) > max_characters:
            break
        sections.append(section)
        citations.append(citation)
        selected.append(result)
        used += len(separator) + len(section)

    if not selected:
        return ContextResult(
            query=query,
            context="",
            citations=(),
            results=(),
            abstained=True,
            reason="supporting result exceeds context budget",
        )
    return ContextResult(
        query=query,
        context="\n\n".join(sections),
        citations=tuple(citations),
        results=tuple(selected),
        abstained=False,
    )


def retrieval_metrics(
    retriever: BM25Retriever,
    queries: list[tuple[str, set[str]]],
    *,
    top_k: int = 5,
    group_by_document: bool = True,
    expand_query: bool = False,
) -> dict[str, float]:
    """Calculate hit rate, recall, and reciprocal rank for labelled query IDs."""

    if not queries:
        raise ValueError("at least one labelled query is required")
    hits = 0
    reciprocal_rank = 0.0
    retrieved_relevant = 0
    total_relevant = 0
    for query, relevant_ids in queries:
        results = retriever.retrieve(
            query,
            top_k=top_k,
            group_by_document=group_by_document,
            expand_query=expand_query,
        )
        result_ids = [result.chunk.document_id for result in results]
        relevant_ids = set(relevant_ids)
        total_relevant += len(relevant_ids)
        retrieved_relevant += len(set(result_ids) & relevant_ids)
        for rank, document_id in enumerate(result_ids, 1):
            if document_id in relevant_ids:
                hits += 1
                reciprocal_rank += 1.0 / rank
                break
    count = len(queries)
    return {
        "queries": float(count),
        "hit_rate_at_k": hits / count,
        "recall_at_k": retrieved_relevant / max(total_relevant, 1),
        "mrr_at_k": reciprocal_rank / count,
    }
