#!/usr/bin/env python3
"""Evaluate GPU dense retrieval with a sentence-transformers encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from llm.rag import BM25Retriever, expand_petroleum_query, load_jsonl_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bm25-weight", type=float, default=0.0)
    args = parser.parse_args()
    if args.top_k <= 0 or args.batch_size <= 0 or args.bm25_weight < 0:
        raise SystemExit("--top-k and --batch-size must be positive")

    chunks = load_jsonl_chunks(args.documents)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    queries = [json.loads(line) for line in args.queries.open(encoding="utf-8") if line.strip()]
    query_embeddings = model.encode(
        [item["query"] for item in queries],
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    scores = query_embeddings @ embeddings.T
    bm25 = BM25Retriever(chunks) if args.bm25_weight else None
    hits = reciprocal = retrieved = total = 0
    details = []
    for row, query_scores in zip(queries, scores):
        if bm25:
            lexical = torch.tensor(
                [bm25.score(expand_petroleum_query(row["query"]), index) for index in range(len(chunks))],
                device=query_scores.device,
            )
            dense_min, dense_max = query_scores.min(), query_scores.max()
            lex_min, lex_max = lexical.min(), lexical.max()
            query_scores = (query_scores - dense_min) / (dense_max - dense_min + 1e-8)
            lexical = (lexical - lex_min) / (lex_max - lex_min + 1e-8)
            query_scores = query_scores + args.bm25_weight * lexical
        ranked = torch.argsort(query_scores, descending=True).tolist()
        selected = []
        seen = set()
        for index in ranked:
            document_id = chunks[index].document_id
            if document_id in seen:
                continue
            selected.append(document_id)
            seen.add(document_id)
            if len(selected) == args.top_k:
                break
        relevant = set(row["relevant_document_ids"])
        overlap = set(selected) & relevant
        retrieved += len(overlap)
        total += len(relevant)
        rank = next((position for position, doc in enumerate(selected, 1) if doc in relevant), None)
        if rank is not None:
            hits += 1
            reciprocal += 1.0 / rank
        details.append({"query": row["query"], "retrieved": selected, "relevant": list(relevant)})
    result = {
        "model": args.model,
        "device": device,
        "chunks": len(chunks),
        "queries": len(queries),
        "top_k": args.top_k,
        "bm25_weight": args.bm25_weight,
        "metrics": {
            "hit_rate_at_k": hits / len(queries),
            "recall_at_k": retrieved / max(total, 1),
            "mrr_at_k": reciprocal / len(queries),
        },
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"device": device, "chunks": len(chunks), **result["metrics"]}, indent=2))
    print(f"Result: {args.output}")


if __name__ == "__main__":
    main()
