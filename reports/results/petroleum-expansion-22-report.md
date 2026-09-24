# Petroleum Corpus Expansion: 22 Sources

Added and checksum-verified one USGS public-domain source:

- `usgs-ofr-2011-1167`
- USGS Methodology for Assessing Continuous Petroleum Resources
- 2011
- 75 PDF pages
- SHA-256: `96f9677123c6f88c3095b47e38288a0415730080c0574d97fca6fd0a1f63bcbe`

## Ingestion

```text
Manifest sources: 22
Recovered pages: 1,944
Original chunks: 4,893
Clean chunks: 4,317
Rejected chunks: 576
Failed pages: 1
```

Rejected pages were isolated by the existing quality filter. The one extraction failure remains a
review item and was not silently treated as clean prose.

## Regression

The fixed 50-question production benchmark was rerun after the expansion:

```text
Hit@5:    0.980
Recall@5: 0.981
MRR@5:    0.761
```

Previous 21-source baseline:

```text
Hit@5:    0.980
Recall@5: 0.981
MRR@5:    0.759
```

The new source does not regress retrieval. It is now included in the manifest and must receive the
same review and provenance treatment as future additions.
