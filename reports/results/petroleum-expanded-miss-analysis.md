# Expanded Petroleum Retrieval Miss Analysis

**Corpus:** `data/petroleum/chunks-expanded-clean.jsonl`  
**Evaluation:** 18 fixed questions, BM25 top-5, pre-chunked loading  
**Result:** 5 misses; hit rate@5 0.722, recall@5 0.737, MRR@5 0.479

## Findings

| # | Query | Classification | Finding | Recommended action |
|---:|---|---|---|---|
| 1 | How does salt act in the petroleum system of East Siberia? | Query vocabulary / passage competition | The target page explicitly says salt is the seal, but the query uses `act` and `petroleum system`; BM25 favors other East Siberia pages. | Add controlled domain synonyms (`role`, `seal`, `evaporite`, `anhydrite`) and test query expansion separately. |
| 2 | What kind of hydrocarbons are considered likely in the southern area of NPR-4? | Table extraction / chunking | The target evidence is in an extracted table and says indigenous hydrocarbons are overmature and limited to dry gas. The table is split across chunks and some parts are rejected as corrupted. | Build a table-aware representation; do not rely on ordinary prose filtering for this passage. |
| 3 | What petroleum resource is mapped in the Piceance Basin oil shale study? | Chunk competition / query wording | Page 7 contains the answer, including 1.53 trillion barrels of oil in place, but several chunks from other pages outrank it. | Aggregate scores by parent page/report before selecting top-k passages; add a numeric-resource query variant. |
| 4 | What was the purpose of the Indian National Gas Hydrate Program Expedition 01? | Missing source coverage / incorrect label | The manifest points to `sir2012-5054_front.pdf`, which contains only front matter (15 extracted pages). The labelled page is a cover page and cannot answer the purpose question. | Obtain the complete report or relabel the question to a fact supported by the front matter. |
| 5 | What potential effects can Fayetteville Shale gas production have on shallow aquifers? | Chunk competition / page-level duplication | The target abstract page contains the answer, but multiple chunks from pages 12 and 38 occupy top-k. | Use parent-page diversity or aggregate page scores; retain the abstract as a high-value answer passage. |

## Parent-page grouping result

Parent-page max-score grouping was implemented in `llm/rag.py` and enabled by default for context
assembly and retrieval metrics. It keeps one representative chunk per `document_id`, preventing
multiple chunks from the same page from consuming top-k.

On the unchanged 18-question benchmark:

- Hit rate@5: **0.722 -> 0.778**
- Recall@5: **0.737 -> 0.789**
- MRR@5: **0.479 -> 0.490**
- Misses: **5 -> 4**

The Piceance Basin miss was resolved by parent-page grouping. The remaining misses are listed above.

## Conclusion

These misses are not evidence that more reports are immediately needed. Two are primarily
retrieval/chunk-ranking problems, one requires table processing, and one is a corpus coverage
problem caused by ingesting only a report front matter file.

## Controlled query variants to evaluate separately

These should be added as a separate variant set, not replace the fixed benchmark:

```text
What role does salt play as a seal in East Siberia?
What hydrocarbons are expected in the southern NPR-4 area?
What oil-in-place resource is described for the Piceance Basin?
How can drilling, hydraulic fracturing, and flowback affect shallow groundwater?
```

The Indian Expedition question should not receive a variant until the complete report is acquired
or the label is corrected.

## Controlled expansion result

A conservative optional expansion was added behind `--expand-petroleum-query`. It preserves the
original query and adds limited terms such as `salt -> seal/evaporite/anhydrite` and
`aquifers -> groundwater/water-quality`.

On the unchanged 18-question benchmark, combined with parent-page grouping:

- Hit rate@5: **0.778 -> 0.833**
- Recall@5: **0.789 -> 0.842**
- MRR@5: **0.490 -> 0.536**

The option is deliberately not enabled by default in the retrieval library, so the original BM25
baseline remains reproducible.

## Next implementation priorities

1. Add parent-page aggregation and a per-page diversity limit to retrieval.
2. Add controlled petroleum query expansion and compare against the unchanged fixed benchmark.
3. Acquire the complete NGHP-01 report and replace the front-matter-only source.
4. Add a separate table/figure extraction path.
