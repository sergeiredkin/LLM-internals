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
First-pass accepted: 84
First-pass rejected:  16
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

The original fixed 10-question set was run against the expanded clean corpus:

```text
Hit rate@5: 0.800
Recall@5:   0.818
MRR@5:      0.683
```

The metrics did not change, which is expected because the fixed questions target the original
reports. The expanded corpus is now large enough to add questions targeting the eight new sources. The
100-sample automated first pass labels 84 samples as prose candidates and 16 as requiring rejection
or confirmation; the generated labels remain local for manual correction.

## Decision

The expansion pipeline is working. Do not claim broad petroleum coverage yet: manually review the
100-sample expansion, add labelled questions for the new topics, then expand toward 50 reports before
introducing dense retrieval or reranking.
