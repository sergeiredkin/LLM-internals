# Gold Answers: 50-Question Petroleum Benchmark

Built 2026-10-05 by `scripts/build_gold_answers.py` from the pipeline drafts in
`data/petroleum/answer-eval-50.jsonl`. Every gold answer is extractive: each sentence is a
verbatim window (similarity >= 0.75) from a cited evidence page, so grounding
holds by construction rather than by trust.

## Status

```text
Verified:    43/50
Needs review: 7
Abstention:   0
Review causes: numeric grounding 1, lexical support 0
```

Records with `status: review` list machine-readable reasons in `review_reasons`
and still require human sign-off before use as gold labels.
