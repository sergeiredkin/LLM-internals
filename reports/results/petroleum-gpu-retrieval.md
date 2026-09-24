# GPU Dense Retrieval Evaluation

Hardware: NVIDIA GeForce RTX 3060 12 GB
Environment: `gpu-test`

## Results

| Method | Hit@5 | Recall@5 | MRR@5 |
|---|---:|---:|---:|
| Dense GPU (`all-MiniLM-L6-v2`) | 0.720 | 0.712 | 0.460 |
| Dense + BM25 fusion (0.25) | 0.800 | 0.788 | 0.553 |
| Dense + BM25 fusion (0.50) | 0.820 | 0.808 | 0.570 |
| Dense + BM25 fusion (1.00) | 0.860 | 0.846 | 0.611 |
| GPU cross-encoder, candidate-k=25 | 0.860 | 0.865 | 0.638 |
| GPU cross-encoder, candidate-k=50 | 0.820 | 0.827 | 0.630 |
| Lexical section/summary/semantic pipeline | **0.980** | **0.981** | **0.759** |

The cross-encoder ran successfully on CUDA, but did not beat the deterministic lexical pipeline.
Candidate-k=25 was better than candidate-k=50, so expanding the candidate pool currently adds noise.

## Decision

Keep GPU dense/cross-encoder retrieval experimental. The production path remains lexical retrieval
with table facts, section-aware ranking, summary-aware ranking, and lightweight semantic reranking.
The GPU evaluator is available at `scripts/evaluate_dense_retrieval.py`; cross-encoder evaluation is
available at `scripts/evaluate_cross_encoder.py`.
