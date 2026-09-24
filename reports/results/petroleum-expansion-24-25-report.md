# Petroleum Corpus Expansion: Sources 24–25

## Source 24

- `usgs-ofr-2010-1309`
- Improved USGS Methodology for Assessing Continuous Petroleum Resources Using Analogs
- USGS, 2010
- 29 pages
- SHA-256: `505faf619bf8e93e1904b1bae726035d83dc3cc36d6335936028fc043e1007f3`

Regression with source 24:

```text
Hit@5:    0.980
Recall@5: 0.981
MRR@5:    0.761
```

No regression.

## Source 25

- `usgs-fs-1997-0145`
- Changing Perceptions of World Oil and Gas Resources as Shown by Recent USGS Petroleum Assessments
- USGS, 1997
- 4 pages
- SHA-256: `67eab3995be4eb6da239ce394d46d3972ab15d1d8ab8b810f2aaeda21121f8f7`

Regression with source 25:

```text
Hit@5:    0.960
Recall@5: 0.962
MRR@5:    0.757
```

The source is checksum-valid and public-domain, but its short high-level content introduces
retrieval noise. It remains in the verified corpus and is marked with `source_role: overview`.
Source-aware weighting now downranks overview material without changing the 50-question regression.

Combined verified manifest size: 25 sources. With source-aware weighting enabled, the production
benchmark returns hit@5 0.980, recall@5 0.981, and MRR@5 0.761.
