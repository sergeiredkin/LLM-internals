import math, torch, matplotlib.pyplot as plt
torch.manual_seed(0)

CFG = dict(
    steps=600, lr=0.05, warmup=50,
    schedule="cosine",        # cosine | linear | constant | inverse_sqrt
    min_lr_frac=0.1,
    betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    cond=100.0,               # curvature ratio of toy problem: f = 0.5*(cond*x^2 + y^2)
    x0=(10.0, 10.0),
)

def lr_at(step, c):
    lr, w = c["lr"], c["warmup"]
    if step < w:
        return lr * (step + 1) / max(w, 1)                    # linear warmup
    t, Tt = step - w, max(c["steps"] - w, 1)
    if   c["schedule"] == "cosine":   f = 0.5 * (1 + math.cos(math.pi * t / Tt))
    elif c["schedule"] == "linear":   f = 1 - t / Tt
    elif c["schedule"] == "constant": f = 1.0
    elif c["schedule"] == "inverse_sqrt": return lr / math.sqrt(1 + t / max(w, 1))
    return c["min_lr_frac"] * lr + (1 - c["min_lr_frac"]) * lr * f

class AdamW:                                                   # the real update, nothing hidden
    def __init__(self, params, lr, betas, eps, wd):
        self.p = list(params)
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, *betas, eps, wd
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.p]
        self.v = [torch.zeros_like(p) for p in self.p]
    @torch.no_grad()
    def step(self, lr):
        self.t += 1
        for p, m, v in zip(self.p, self.m, self.v):
            g = p.grad
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)         # 1st moment
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)  # 2nd moment
            mh = m / (1 - self.b1 ** self.t)                   # bias correction
            vh = v / (1 - self.b2 ** self.t)
            if self.wd: p.sub_(p, alpha=lr * self.wd)          # DECOUPLED weight decay
            p.sub_(mh / (vh.sqrt() + self.eps), alpha=lr)      # the Adam step

def run(c):
    q = torch.tensor(c["x0"], dtype=torch.float64).requires_grad_(True)
    opt = AdamW([q], c["lr"], c["betas"], c["eps"], c["weight_decay"])
    losses = []
    for s in range(c["steps"]):
        loss = 0.5 * (c["cond"] * q[0] ** 2 + q[1] ** 2)
        q.grad, = torch.autograd.grad(loss, q)
        opt.step(lr_at(s, c))
        losses.append(min(loss.item(), 1e4))
        if loss.item() > 1e12 or loss.item() != loss.item():
            print(f"  DIVERGED at step {s}"); break
    return losses, q.detach()

runs = {
    "cosine (default)":    {},
    "constant":            {"schedule": "constant"},
    "linear":              {"schedule": "linear"},
    "no warmup, constant": {"schedule": "constant", "warmup": 0},
    "lr=2.0 (too high)":   {"lr": 2.0},
    "eps=1e-1 (stall)":    {"eps": 1e-1},
    "wd=1e-1":             {"weight_decay": 1e-1},
}
plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
for name, over in runs.items():
    c = {**CFG, **over}
    losses, q = run(c)
    plt.plot(losses, label=f"{name} [{losses[-1]:.1e}]")
    print(f"{name:22s} final loss {losses[-1]:.3e}  params {q.tolist()}")
plt.yscale("log"); plt.legend(fontsize=8); plt.title("loss")
plt.subplot(1, 2, 2)
for name, over in runs.items():
    c = {**CFG, **over}
    plt.plot([lr_at(s, c) for s in range(c["steps"])], label=name)
plt.title("LR(t)"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()
print("\nLR preview (cosine):", {s: round(lr_at(s, CFG), 4) for s in [0, 25, 49, 50, 150, 300, 599]})

# TRY:
# - lr=2.0: loss plateaus ~200 — Adam's per-coord step is ~lr, so x bounces ±2 forever.
#   This is "training exploded into oscillation", the most common real-world failure.
# - eps=1e-1: near the optimum v̂->0 so the update ~ lr*m̂/eps — watch it stall 2-3 decades early.
# - wd=1e-1: decoupled decay shrinks params toward 0 -> higher loss floor. Compare with wd applied
#   inside the gradient (L2) by hand to see why decoupling differs.
# - betas=(0.9, 0.5): v is a short average -> noisy effective LR -> jittery loss.
# - schedule="inverse_sqrt" (the original Transformer schedule).
# - cond=1e4, warmup=0, lr=0.5: this toy can't fully reproduce LLM blowups (they come from rare
#   huge-gradient tokens hitting a stale v), but the LR-shape intuition transfers 1:1.