# Petroleum RAG Parallel Sprint

Date: 2026-09-24

## Retrieval comparison

All runs used the fixed 18-question set, parent-page grouping, controlled query expansion, and top-k=5.

| Configuration | Hit@5 | Recall@5 | MRR@5 |
|---|---:|---:|---:|
| Clean prose BM25 | 0.889 | 0.850 | 0.592 |
| Prose + table facts | 0.944 | 0.900 | 0.578 |
| Prose + table facts + primary rerank | 0.889 | 0.850 | 0.592 |

The augmented corpus is the recall configuration. The primary-evidence rerank restores the baseline
MRR and remains the safer answer-generation configuration until a stronger reranker is evaluated.

## Answer evaluation

A draft extractive answer set was generated for all 18 questions with citations and source evidence.
The evaluator returned 1.000 for citation coverage, lexical support, numeric grounding, abstention,
and grounded rate.

These results are a pipeline smoke test, not a final answer-quality score: the draft answers are
constructed from the cited evidence itself. The draft file must be manually rewritten into concise,
independent gold answers before it is used as a quality benchmark.

## Table validation

The table pipeline produced 128 conservative fact candidates. The NPR-4 Table 4 group preserves
pages 55–57 and includes the fact that the southern play is limited to dry gas. Facts retain source,
page range, extraction method, and medium confidence.

## Decision

Use primary-evidence reranking for answer generation. Keep table facts enabled as a diagnostic and
recall layer. Do not expand the corpus or start QLoRA until the manually reviewed answer set exists.
