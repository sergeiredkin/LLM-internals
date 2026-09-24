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
| 4 | What was the purpose of the Indian National Gas Hydrate Program Expedition 01? | Newly covered source / chunk competition | A checksum-verified 34-page Expedition Summary was added. Its abstract on page 5 and project goal on page 7 support the question; page 5 is the highest-ranked answer passage. | Retain both page 5 and page 7 as relevant labels and improve section-aware ranking for summary/goal passages. |
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

These misses are not evidence that more reports are immediately needed. Three are primarily
retrieval/chunk-ranking problems, one requires table processing, and the previous corpus coverage
problem for the Indian Expedition question has been corrected by adding the Expedition Summary.

## Controlled query variants to evaluate separately

These should be added as a separate variant set, not replace the fixed benchmark:

```text
What role does salt play as a seal in East Siberia?
What hydrocarbons are expected in the southern NPR-4 area?
What oil-in-place resource is described for the Piceance Basin?
How can drilling, hydraulic fracturing, and flowback affect shallow groundwater?
```

The Indian Expedition question now has supported page-5 and page-7 labels in the added Expedition
Summary. The abstract page is the best direct answer passage.

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
4. Add a separate table/figure extraction path. The first table candidate corpus now merges labelled continuation pages; NPR-4 Table 4 pages 55–57 form one table group and the isolated NPR-4 evaluation retrieves the relevant group at hit rate@5 1.000.
