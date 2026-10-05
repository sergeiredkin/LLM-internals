#!/usr/bin/env python3
"""Ask the petroleum RAG a question through a local Ollama LLM.

Retrieval uses the frozen production configuration (lexical-first BM25 with
query expansion, evidence preference, section/summary awareness, semantic
rerank, and source-aware weighting). The LLM answers ONLY from the retrieved
sources and must cite them; the reply is validated for citation coverage and
numeric grounding against the same evidence.

Examples:
    # One question:
    conda run -n gpu-test python -m scripts.ask_petroleum_ollama \
        --question "What formations are proven reservoirs in the Amerasia Basin?"

    # Run the whole gold set end-to-end:
    conda run -n gpu-test python -m scripts.ask_petroleum_ollama \
        --questions data/petroleum/gold-answers-50.jsonl \
        --output reports/results/petroleum-ollama-50q.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

from llm.answer_quality import evaluate_answer
from llm.rag import BM25Retriever, load_jsonl_chunks

DEFAULT_DOCUMENTS = Path("data/petroleum/chunks-augmented.jsonl")
DEFAULT_MODEL = "qwen3.5:latest"
DEFAULT_URL = "http://localhost:11434"
CITE_RE = re.compile(r"\[([^\[\]]{1,40})\]")
SOURCE_CITE_RE = re.compile(r"\bSource\s+(\d{1,2})\b", re.IGNORECASE)


def parse_citations(reply: str) -> set[int]:
    """Extract source indexes from bracket groups like [1], [1, 3], [2: page-0034],
    [1, p. 106]. Leading integers of comma/colon-separated items count; page
    annotations ('p. 106') survive parsing but are later bounds-filtered."""
    indexes: set[int] = set()
    for group in CITE_RE.findall(reply):
        for token in re.split(r"[,;]", group):
            match = re.match(r"\s*(\d{1,2})\b", token)
            if match:
                indexes.add(int(match.group(1)))
    for n in SOURCE_CITE_RE.findall(reply):
        indexes.add(int(n))
    return indexes
NOT_FOUND = "not found in the verified corpus"
MAX_SOURCE_CHARS = 1400

SYSTEM_PROMPT = (
    "You are a precise petroleum-geology research assistant. Answer ONLY using the "
    "numbered SOURCES provided in the user message. Cite every claim inline with its "
    "source number, like [1] or [2]. If some part of the question is answerable from "
    "the sources, answer that part and cite it. Reply exactly: Not found in the verified "
    "corpus. - only when the question is entirely unanswerable from the sources. Never "
    "use outside knowledge, never guess numbers, and keep all figures and units exactly "
    "as written in the sources."
)


def ollama_chat(url: str, model: str, question: str, sources: str, timeout: int) -> tuple[str, float]:
    payload = json.dumps(
        {
            "model": model,
            "stream": False,
            "think": False,  # qwen3+ reasoning mode burns hidden tokens; not needed for extractive QA
            "options": {"temperature": 0.1, "num_predict": 512, "num_ctx": 8192},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"QUESTION: {question}\n\nSOURCES:\n{sources}"},
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{url}/api/chat", data=payload, headers={"Content-Type": "application/json"}
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    return str(body.get("message", {}).get("content", "")).strip(), time.monotonic() - started


def format_sources(hits: list) -> tuple[str, list[str]]:
    # The production retriever can return several chunks from the same parent
    # page; keep the best-scored chunk per document_id so sources (and the
    # citation map) are unambiguous.
    seen: dict[str, object] = {}
    for result in hits:
        doc_id = result.chunk.document_id
        if doc_id not in seen:
            seen[doc_id] = result
    unique = list(seen.values())
    blocks, doc_ids = [], []
    for index, result in enumerate(unique, 1):
        chunk = result.chunk
        text = " ".join(chunk.text.split())[:MAX_SOURCE_CHARS]
        blocks.append(f"[{index}] id={chunk.document_id}\n{text}")
        doc_ids.append(chunk.document_id)
    return "\n\n".join(blocks), doc_ids


def ask(retriever, question: str, args) -> dict:
    hits = retriever.retrieve(
        question,
        top_k=args.top_k,
        expand_query=True,
        prefer_primary_evidence=True,
        section_aware=True,
        summary_aware=True,
        semantic_rerank=True,
        source_aware=True,
    )
    sources_text, doc_ids = format_sources(hits)
    raw_reply, latency = ollama_chat(args.url, args.model, question, sources_text, args.timeout)

    cited_indexes = parse_citations(raw_reply)
    cited_ids = [doc_ids[n - 1] for n in sorted(cited_indexes) if 1 <= n <= len(doc_ids)]

    # Hedging models answer correctly and still append the not-found boilerplate;
    # an abstention only counts when there are no citations backing the reply.
    reply = raw_reply
    if cited_ids and NOT_FOUND.lower() in reply.lower():
        pattern = re.compile(r"[^.!?]*" + re.escape(NOT_FOUND) + r"[^.!?]*[.!?]?", re.IGNORECASE)
        reply = pattern.sub("", reply).strip()
    abstained = not cited_ids and NOT_FOUND.lower() in raw_reply.lower()
    checks = evaluate_answer(
        {
            "query": question,
            "answer": "" if abstained else raw_reply,
            "citations": cited_ids,
            "evidence": [{"document_id": d, "text": r.chunk.text} for d, r in zip(doc_ids, hits)],
            "expected_abstention": False,
        }
    )
    return {
        "query": question,
        "reply": reply,
        "raw_reply": raw_reply,
        "citations": cited_ids,
        "retrieved_ids": doc_ids,
        "abstained": abstained,
        "latency_seconds": round(latency, 2),
        "checks": checks,
        "cited_retrieved": bool(cited_ids) or abstained,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--question")
    parser.add_argument("--questions", type=Path, help="JSONL with a 'query' field (e.g. gold answers)")
    parser.add_argument("--limit", type=int, default=0, help="only run the first N questions")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--show-sources", action="store_true")
    args = parser.parse_args()

    if not args.question and not args.questions:
        parser.error("provide --question or --questions")

    chunks = load_jsonl_chunks(args.documents)
    retriever = BM25Retriever(chunks)
    print(f"corpus: {len(chunks)} chunks | model: {args.model} | url: {args.url}")

    questions = [args.question] if args.question else []
    if args.questions:
        for line in args.questions.open(encoding="utf-8"):
            if line.strip():
                questions.append(json.loads(line)["query"])
        if args.limit:
            questions = questions[: args.limit]

    results = []
    handle = args.output.open("w", encoding="utf-8") if args.output else None
    try:
        for index, question in enumerate(questions, 1):
            result = ask(retriever, question, args)
            results.append(result)
            if handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
            flag = "ABSTAIN" if result["abstained"] else ("ok" if result["cited_retrieved"] else "NO-CITE")
            print(f"[{index}/{len(questions)}] {flag} {result['latency_seconds']:>5.1f}s  {question[:70]}")
    finally:
        if handle:
            handle.close()

    if results and len(results) > 1:
        cited = sum(1 for r in results if r["cited_retrieved"])
        abstained = sum(1 for r in results if r["abstained"])
        grounded = sum(1 for r in results if r["checks"]["grounded"])
        avg = sum(r["latency_seconds"] for r in results) / len(results)
        print(
            f"\nsummary: {cited}/{len(results)} cited|abstained correctly, "
            f"{grounded} fully grounded, {abstained} abstentions, avg {avg:.1f}s"
        )
        if args.output:
            print(f"report: {args.output}")

    if args.question:
        result = results[0]
        print(f"\n{result['reply']}")
        if result["citations"]:
            print(f"\ncitations: {', '.join(result['citations'])}")
        if args.show_sources:
            print("\nretrieved:")
            for doc_id in result["retrieved_ids"]:
                print(f"  - {doc_id}")


if __name__ == "__main__":
    main()
