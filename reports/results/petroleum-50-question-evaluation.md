# Petroleum 50-Question Evaluation

## Dataset

`data/petroleum/queries-50.jsonl` contains the original 18 questions unchanged plus 32 expansion
questions. The expansion covers geology, petroleum systems, assessments, resources, gas hydrates,
and numerical/unit questions. Relevant page IDs are preserved for every question.

## Retrieval results

All runs use pre-chunked data, parent-page grouping, controlled query expansion, and top-k=5.

| Configuration | Hit@5 | Recall@5 | MRR@5 |
|---|---:|---:|---:|
| Clean BM25 | 0.820 | 0.808 | 0.603 |
| Prose + table facts | 0.860 | 0.846 | 0.595 |
| Prose + table facts + primary rerank | 0.840 | 0.827 | 0.600 |

The expanded 50-question set is harder than the original set, as expected. The table-fact layer
improves recall, while primary reranking improves MRR and reduces some fact-induced ranking noise.

## Interpretation

- The original 18-question benchmark remains the regression gate.
- The 50-question benchmark is now the development benchmark.
- Do not alter the original labels to improve scores.
- Before corpus expansion, inspect the 50-question misses and manually validate all new labels.
- Dense/hybrid retrieval is the next engineering comparison.
