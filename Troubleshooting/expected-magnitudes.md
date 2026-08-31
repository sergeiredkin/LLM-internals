---
type: moc
---
# Expected Magnitudes (the silent-bug detector)
| Quantity | Healthy | Alarm means |
|---|---|---|
| f64 manual-vs-autograd | 1e-12…1e-17 | 1e-2 broken chain rule; ~20 broken CHECKER |
| f32 checker | ~1e-5 | 1e-1 real error |
| LM loss at init | ln(V)≈10.8 (50k vocab) | above = worse than random = bug |
| KV vs no-cache | 1e-6…1e-5 | ~1e-1 = mask/slicing bug |
| quant rel-err | 0…1 | >1 scale collapsed (outliers/tensor mode) |
| RMSNorm per-token RMS | 1.0000 | ≠1 → eps or shape bug |
| LS loss floor | formula (script 08) | flat AT floor = done, not broken |
