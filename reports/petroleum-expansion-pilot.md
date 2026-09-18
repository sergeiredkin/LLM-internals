# Petroleum Corpus Expansion Pilot

## Expansion

The verified USGS manifest grew from 12 to **20 reports**. The eight added reports cover coalbed gas,
gas hydrates, oil shale, Cook Inlet, Austin Chalk oil and gas, and Fayetteville Shale gas.

All added PDFs were downloaded from official USGS publication URLs and checksum-verified before
extraction.

## Result

```text
Reports:          20
Recovered pages:  1,838
Original chunks:  4,613
Clean chunks:     4,069
Rejected chunks:    544
Review sample:      100
Reviewed accepted:   84
Reviewed rejected:    16
```

Rejection flags include overlapping categories:

```text
figure/map-only: 384
short:            279
corrupted text:    21
sparse symbols:    54
bibliography:       2
```

The previously malformed South America page remains represented as a caption-only figure record.

## Retrieval comparison

The expanded 18-question set adds one labelled question for each new report:

```text
Hit rate@5: 0.722
Recall@5:   0.737
MRR@5:      0.479
```

The lower score is useful: the new topics expose retrieval weaknesses before further corpus growth.
Misses are recorded in `reports/results/petroleum-bm25-expanded-misses.json`.

## Decision

The 100-sample review is confirmed: 84 samples are prose candidates and 16 are rejected as figures,
charts, bibliography, title material, or unusable symbol extraction. Decisions are reproducible through
`data/petroleum/review-expanded-overrides.json`.

Do not expand toward 50 reports yet. First inspect the five expanded-query misses and improve query
coverage/chunk retrieval against this 20-report corpus.
