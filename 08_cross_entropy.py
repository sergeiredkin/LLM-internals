import math, torch, torch.nn.functional as F
torch.manual_seed(0)

CFG = dict(V=1000, B=8, label_smoothing=0.1, logit_scale=1.0)
V, B = CFG["V"], CFG["B"]
dt = torch.float64
logits = torch.randn(B, V, dtype=dt) * CFG["logit_scale"]
targets = torch.randint(0, V, (B,))
ar = torch.arange(B)

print("--- 1. numerical stability (fp64 doesn't save the naive version) ---")
for s in (1, 100, 1000):
    lg = logits * s
    naive = torch.log(torch.softmax(lg, -1))[ar, targets].mean()
    stable = torch.log_softmax(lg, -1)[ar, targets].mean()
    print(f"  scale={s:5d}  naive {naive.item():>8.3f}   log_softmax {stable.item():8.3f}   "
          f"F.cross_entropy {F.cross_entropy(lg.float(), targets).item():8.3f}")

print("\n--- 2. the gradient IS (softmax - onehot)/B ---")
lg = logits.clone().requires_grad_(True)
F.cross_entropy(lg, targets).backward()
p = logits.softmax(-1)
onehot = F.one_hot(targets, V).to(dt)
g_manual = (p - onehot) / B
print(f"  max |autograd - manual| = {(lg.grad - g_manual).abs().max().item():.2e}")
print("  -> big loss gradient = wrong class with high prob; ~0 = confident & correct.")

print("\n--- 3. label smoothing: loss floor ---")
eps = CFG["label_smoothing"]
def ls_ce(l, t, eps):
    ls = l.log_softmax(-1)
    return -((1 - eps) * ls[ar, t] + eps * ls.mean(-1)).mean()
print(f"  manual LS CE {ls_ce(logits, targets, eps).item():.4f}   "
      f"F.cross_entropy(LS) {F.cross_entropy(logits.float(), targets, label_smoothing=eps).item():.4f}")
q_true, q_other = 1 - eps + eps / V, eps / V
floor = -(q_true * math.log(q_true) + (V - 1) * q_other * math.log(q_other))
print(f"  PERFECT-model floor with eps={eps}: {floor:.4f}  (plain CE floor = 0)")
print(f"  -> if your smoothed training loss flattens here, the model is DONE, not broken.")

print("\n--- 4. ignore_index & class weights ---")
t_pad = targets.clone(); t_pad[-2:] = -100
loss = F.cross_entropy(logits.float(), t_pad, ignore_index=-100)
sel = t_pad != -100
manual = -logits.log_softmax(-1)[sel, t_pad[sel]].mean()
print(f"  ignore_index: F {loss.item():.4f} = manual over non-ignored {manual.item():.4f} "
      f"(mean over {int(sel.sum())} rows, not {B})")
w = torch.ones(V); w[0] = 50.0
print(f"  class weight 50 on class 0 -> loss moves: "
      f"{F.cross_entropy(logits.float(), targets, weight=w).item():.4f} vs "
      f"{F.cross_entropy(logits.float(), targets).item():.4f}")

print("\n--- 5. reading loss curves (V=50257) ---")
print(f"  uniform/random init loss = ln(V) = {math.log(50257):.2f}  -> ppl {50257:,}")
for L in (7.0, 4.0, 3.0, 2.0, 1.5):
    print(f"  loss {L:5.2f} -> ppl {math.exp(L):8,.1f}")
print("  loss ABOVE ln(V) at init = worse than random = bug or label noise.")

# TRY:
# - logit_scale=50, scale sweep in section 1: naive -> -inf, log_softmax unaffected. This is why
#   "always use F.cross_entropy / log_softmax" and never log(softmax(x)).
# - label_smoothing=0.5: floor is huge; also note optimal p_true < 1 -> smoothing caps logit gaps
#   (that's the regularization).
# - B=1 vs B=8 in section 2: gradient scale is exactly 1/B — batch size is baked into the loss.
# - Recompute section 2 with logits*100: gradient shrinks (softmax saturated) -> CE gradients die
#   on overconfident wrong predictions. Connects to loss spikes after LR spikes.