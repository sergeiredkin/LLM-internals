# Petroleum Corpus Expansion: 23 Sources

Added and checksum-verified:

- `usgs-ds-547`
- Improved USGS Methodology for Assessing Continuous Petroleum Resources
- USGS, 2012
- 30 PDF pages
- SHA-256: `90975ac43c4aeefb88b7cb079a5deead8ab36692b34b9ca82a30c09959d80427`

## Ingestion

```text
Manifest sources: 23
Recovered pages: 1,971
Original chunks: 4,965
Clean chunks:    4,382
Rejected chunks: 583
Failed pages:    1
```

## Regression

The frozen 50-question production benchmark remains stable:

```text
Hit@5:    0.980
Recall@5: 0.981
MRR@5:    0.761
```

The source passed checksum validation, extraction, quality filtering, and retrieval regression.
