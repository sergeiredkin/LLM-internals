#!/usr/bin/env python3
"""Evaluate a GPU cross-encoder reranker over BM25 candidate passages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

from llm.rag import BM25Retriever, expand_petroleum_query, load_jsonl_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--candidate-k", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if min(args.candidate_k, args.top_k, args.batch_size) <= 0:
        raise SystemExit("candidate-k, top-k, and batch-size must be positive")

    chunks = load_jsonl_chunks(args.documents)
    bm25 = BM25Retriever(chunks)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(args.model, device=device)
    queries = [json.loads(line) for line in args.queries.open(encoding="utf-8") if line.strip()]
    hits = reciprocal = retrieved = total = 0
    details = []
    for row in queries:
        query = expand_petroleum_query(row["query"])
        candidates = bm25.retrieve(query, top_k=args.candidate_k, group_by_document=True)
        pairs = [(query, result.chunk.text) for result in candidates]
        rerank_scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)
        ranked = sorted(zip(candidates, rerank_scores), key=lambda item: (-float(item[1]), item[0].chunk.chunk_id))
        selected = [item[0].chunk.document_id for item in ranked[:args.top_k]]
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
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
        "chunks": len(chunks),
        "queries": len(queries),
        "metrics": {
            "hit_rate_at_k": hits / len(queries),
            "recall_at_k": retrieved / max(total, 1),
            "mrr_at_k": reciprocal / len(queries),
        },
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"device": device, **result["metrics"]}, indent=2))
    print(f"Result: {args.output}")


if __name__ == "__main__":
    main()
