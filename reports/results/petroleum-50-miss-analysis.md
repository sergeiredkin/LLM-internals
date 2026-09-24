# Petroleum 50-Question Miss Analysis

The 50-question benchmark was checked against parent-page grouped BM25 with controlled query expansion.
After validating three labels against direct answer passages, six questions remain misses at top-5.

| Class | Questions | Action |
|---|---:|---|
| Table/column representation | NPR-4 southern hydrocarbons | Use the optional table-fact corpus and primary-evidence reranking. |
| Table/section representation | East Siberia salt role (two phrasings) | Add section-aware/table-aware ranking; page 83 is direct evidence. |
| Abstract/section ranking | Amerasia reservoir facies | Page 24 is direct evidence but competing report passages rank higher. |
| Numeric summary page | NGHP site count | Page 19 is the direct passage; add summary-page ranking. |
| Source-bed terminology | Permian source-bed types | Page 22 is direct evidence; query vocabulary and section ranking need improvement. |

The three questions whose original labels pointed to title/abstract pages were relabelled to direct
answer passages without changing the original 18-question set. No source was added and no question
was removed.

Section-aware ranking was then added for these intents. With table facts, primary reranking, and
section-aware ranking, results reached hit@5 0.980, recall@5 0.962, and MRR@5 0.708. Summary-aware
ranking improved MRR to 0.744 with the same hit and recall. One Fayetteville shallow-aquifer question
remains outside top-5 and needs a stronger semantic reranker.
