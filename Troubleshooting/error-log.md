---
type: moc
---
# Error Log
Add rows chronologically. Promote to an inc-* note ([[tpl-incident]]) when the lesson is reusable.
| # | File | Symptom | Mode | Root cause → Fix | Lesson |
|---|---|---|---|---|---|
| 1 | 09 | double != float | crash | init() fp32 vs data fp64; matmul dtype-strict → dtype= in constructors | matmul ≠ elementwise: no silent promotion |
| 2 | 09 | int has no 'mean' | crash | (y-tt)**2 .mean() = (y-tt)**(2.mean()) → parenthesize | attr access binds tighter than ** |
| 3 | 09 | size 8 vs 16 at residual | crash | d_in→width change; skip impossible → layer0 = stem | why GPT blocks preserve d_model |
| 4 | 03 | 257 vs 256 | crash | cache path has +1 pred; x[:n] slices dim0 (no-op at B=1) → x[:,:n] | broadcasting skips size-1 dims silently |
| 5 | 06 | minutes on CPU | perf | NF4 broadcast (N,64,16) ~0.8GB temp → 16-pass scan | trade flops for bandwidth, never reverse |
| 6 | 06 | no plot, no finish | block | plt.show() GUI loop, window behind terminal → Agg | blocking ≠ crash; waiting on YOU |
| 7 | sys | RAM held after stop | system | Ctrl+Z pauses (holds RAM); freed mem returns only at exit | Ctrl+Z ≠ stop |
| 8 | 06 | ended at --- plot --- | block | incomplete patch merge: show survived, Agg absent → MPLBACKEND=Agg | last printed line brackets the hang |
| 9 | 06 | KeyboardInterrupt in hist | success | Ctrl+C snapshot = free stack trace of a hang | the probe worked |
| 10 | term | cannot open file | env | cwd was ~; project elsewhere → cd "path with spaces" | relative-path errors describe YOUR location |
| 11 | 10 | dg check = 2.16e+01 | silent | checker did .sum(0) on (4,8,d) → broadcast garbage → .sum((0,1)) | the checker is code too |
| 12 | 10 | layer_norm TypeError | crash | gain passed into normalized_shape slot (tuple of ints) → adapter lambda | same concept ≠ same signature |
| 13 | 06 | chart ≠ table (0.7179 vs 0.7632) | silent | plot fn ignored its W, built fresh 512² matrix → pass Wo in | duplicated numbers must match; unused param is the fingerprint; tensor/channel err = extreme-value, grouped = LLN |

Near-misses: 09 noise targets (nothing to learn) · 06 lost layer-output metric ·
matplotlib.use before pyplot import · 03_KV_Cash.py naming (grep misses it).
