# Petroleum Corpus Expansion: Source 26

Added and checksum-verified:

- `usgs-osti-6652871`
- Petroleum and Mineral Resources of Antarctica
- USGS/OSTI, 1983
- 82 pages
- SHA-256: `34513183d12853f556d84df3ec7d41356a2e46fd95182181385abb980c43436f`

## Ingestion

```text
Manifest sources: 26
Recovered pages: 2,082
Original chunks: 5,091
Clean chunks:    4,555
Rejected chunks: 808
Failed pages:    1
```

The source has unusually many corrupted/figure-only pages because it is an older scanned report;
these were rejected rather than mixed into prose retrieval.

## Regression

With source-aware weighting enabled:

```text
Hit@5:    0.980
Recall@5: 0.981
MRR@5:    0.761
```

No regression. The source remains in the verified manifest, with rejected extraction artifacts
isolated for possible OCR work later.
