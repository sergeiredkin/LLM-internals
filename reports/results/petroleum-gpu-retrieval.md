# GPU Dense Retrieval Evaluation

Hardware: NVIDIA GeForce RTX 3060 12 GB
Environment: `gpu-test`
Encoder: `sentence-transformers/all-MiniLM-L6-v2`
Corpus: 4,403 chunks including conservative table facts
Benchmark: fixed 50 questions, top-k=5, parent-page grouping

| Method | Hit@5 | Recall@5 | MRR@5 |
|---|---:|---:|---:|
| Dense GPU | 0.720 | 0.712 | 0.460 |
| Dense + BM25 normalized fusion (0.25) | 0.800 | 0.788 | 0.553 |
| Dense + BM25 normalized fusion (0.50) | 0.820 | 0.808 | 0.570 |
| Dense + BM25 normalized fusion (1.00) | 0.860 | 0.846 | 0.611 |
| Lexical section/summary/semantic pipeline | **0.980** | **0.981** | **0.759** |

## Decision

The GPU dense encoder is functional, but it is not competitive on this small domain corpus.
The best dense fusion remains below the lexical pipeline. Keep dense retrieval as an experimental
candidate-generation signal, not the production retriever. The RTX 3060 was successfully used for
embedding inference; peak utilization was transient and memory remained within the 12 GB budget.

The evaluator is available at `scripts/evaluate_dense_retrieval.py`. Its dependency is isolated in
`requirements-rag-gpu.txt` and does not alter the core dependency-free BM25 path.
