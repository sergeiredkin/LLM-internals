---
type: script
number: 3
topic: kv cache
status: done
files: ["03_KV_Cash.py"]
---
## The click
Cache turns per-token decode from O(T^3) total re-forward into O(T^2): counters showed
~153x fewer token-passes, ~400x fewer score elements. GQA (n_kv_heads) halves cache memory.
## Acceptance
cache vs no-cache diff = 1e-6..1e-5 (NOT 0 — different BLAS shapes). Above ~1e-3 = bug (usually mask).
## Incidents
- `gen_c[:T]` sliced the BATCH dim (silent no-op at B=1) -> fixed `gen_c[:, :T]`. [[error-log]]
- filename says "Cash" — the money kind; rename someday (grep misses it).
## Concepts taught
[[kv-cache]]
## Experiments
