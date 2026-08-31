---
type: moc
---
# Seed-0 Regression Baselines
Every rerun must reproduce; deviations = found something.
| Script | Baseline | Criterion |
|---|---|---|
| 06 granularity | .1963/.1430/.1265/.1077/.0971 | bit-identical |
| 06 bits | int8 .0059 / int4 .1077 / int2 .7178 | bit-identical |
| 06 outliers | .7632/.4769/.2002/.1311 | table = chart exactly |
| 06 main | wt .1077 · max-err .337 · layer-out .1075 | layer-out ≈ weight err |
| 09 sec1 | 2.78e-17 (both residual modes) | ≤1e-12 |
| 09 sec3 | no-res ~0.9–1.0 vs resid ~0.2–0.5 | gap widens at depth 32 |
| 10 | sec2 RMS ~1e-6/LN ~1e-5 · sec3 4.501/0.980 vs 0/0 · dg ~1e-5 | sec5 ≤1e-4 |
| 03 | diff 1e-6–1e-5 · counters ~153x/~400x | diff ≤1e-3 |

Scripts 01,02,04,05,07,08: baseline rows added after first successful run.
