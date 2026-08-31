# 09 — Backprop (hand-rolled through an MLP; vanishing/exploding; residuals)
# PATCHED v3: residual now starts at layer 1 — layer 0 maps d_in->width, so it's a stem
#             (no identity skip possible). This is why GPT blocks all preserve d_model.

import math, torch, matplotlib.pyplot as plt
torch.manual_seed(0)

CFG = dict(B=64, d_in=8, width=16, depth=4, act="tanh",   # tanh | relu
           init_gain=1.0, residual=False)

def init(depth, width, d_in, gain, dtype=torch.float64):
    Ws = [torch.randn(width, fan, dtype=dtype) * (gain / math.sqrt(fan))
          for fan in [d_in] + [width] * (depth - 1)]
    bs = [torch.zeros(width, dtype=dtype) for _ in range(depth)]
    Wout = torch.randn(1, width, dtype=dtype) / math.sqrt(width)
    return Ws, bs, Wout

def forward(x, Ws, bs, Wout, act, residual):
    f = torch.tanh if act == "tanh" else torch.relu
    zs, aa = [], []
    a = x
    for l, (W, b) in enumerate(zip(Ws, bs)):
        z = a @ W.T + b; h = f(z)
        zs.append(z)
        a = (a + h) if (residual and l > 0) else h     # PATCHED: l=0 is a stem, no skip
        aa.append(a)
    return aa[-1] @ Wout.T, zs, aa

def manual_backward(y, t, Ws, Wout, zs, aa, x, act, residual):
    B = t.shape[0]
    dy = 2 * (y - t) / B
    gWout = dy.T @ aa[-1]
    da = dy @ Wout
    dWs = [None] * len(Ws)
    for l in reversed(range(len(Ws))):
        a_prev = x if l == 0 else aa[l - 1]
        dz = da * (1 - torch.tanh(zs[l]) ** 2 if act == "tanh" else (zs[l] > 0).double())
        dWs[l] = dz.T @ a_prev
        da = (da + dz @ Ws[l]) if (residual and l > 0) else dz @ Ws[l]  # PATCHED: stem has no identity path
    return dWs, gWout

c = CFG
x = torch.randn(c["B"], c["d_in"], dtype=torch.float64)
t = torch.randn(c["B"], 1, dtype=torch.float64)

print("--- 1. manual vs autograd (float64) ---")
for residual in (False, True):
    Ws, bs, Wout = init(c["depth"], c["width"], c["d_in"], c["init_gain"])
    y, zs, aa = forward(x, Ws, bs, Wout, c["act"], residual)
    dWs, gWout = manual_backward(y, t, Ws, Wout, zs, aa, x, c["act"], residual)
    Ps = [p.clone().requires_grad_(True) for p in Ws] + \
         [Wout.clone().requires_grad_(True)]
    ya, _, _ = forward(x, Ps[:c["depth"]], bs, Ps[-1], c["act"], residual)
    ((ya - t) ** 2).mean().backward()
    diffs = [((g - p.grad).abs().max().item()) for g, p in zip(dWs + [gWout], Ps)]
    print(f"  residual={residual}: max |manual-autograd| over all layers = {max(diffs):.2e}")

print("\n--- 2. gradient norms vs depth (the vanishing/exploding show) ---")
def grad_norms(depth, act, gain, residual):
    xg = torch.randn(64, c["d_in"], dtype=torch.float64); tg = torch.randn(64, 1, dtype=torch.float64)
    Ws, bs, Wout = init(depth, 32, c["d_in"], gain)
    y, zs, aa = forward(xg, Ws, bs, Wout, act, residual)
    dWs, _ = manual_backward(y, tg, Ws, Wout, zs, aa, xg, act, residual)
    return [w.norm().item() for w in dWs]

configs = [("tanh g=0.5", "tanh", 0.5, False), ("tanh g=1.0", "tanh", 1.0, False),
           ("tanh g=3.0", "tanh", 3.0, False), ("relu g=0.5", "relu", 0.5, False),
           ("relu g=sqrt2", "relu", math.sqrt(2), False),
           ("tanh g=1.0 +resid", "tanh", 1.0, True)]
D = 24
plt.figure(figsize=(7, 4))
for name, act, g, res in configs:
    n = grad_norms(D, act, g, res)
    plt.semilogy(range(D), n, marker=".", label=name)
    print(f"  {name:18s} first-layer grad {n[0]:9.2e}   last-layer {n[-1]:9.2e}")
plt.xlabel("layer (0 = input side)"); plt.ylabel("||dW||")
plt.legend(fontsize=8); plt.title(f"grad norms at init, depth={D}")
plt.tight_layout()
plt.savefig("09_grad_norms.png", dpi=120)
plt.show()

print("\n--- 3. tiny training: residual vs not (depth=16, tanh) ---")
def train(residual, steps=600, lr=0.5):
    Ws, bs, Wout = init(16, c["width"], c["d_in"], 1.0)
    Wt = torch.randn(1, c["d_in"], dtype=torch.float64)    # fixed random target function
    xt = torch.randn(256, c["d_in"], dtype=torch.float64)
    tt = torch.tanh(xt @ Wt.T)                             # targets = f(x): there IS signal
    for s in range(steps):
        y, zs, aa = forward(xt, Ws, bs, Wout, "tanh", residual)
        dWs, gWout = manual_backward(y, tt, Ws, Wout, zs, aa, xt, "tanh", residual)
        for w, g in zip(Ws, dWs): w -= lr * g
        Wout -= lr * gWout
        if s % 200 == 0:
            mse = ((y - tt) ** 2).mean()
            print(f"    residual={residual} step {s:4d} mse {mse:.5f}")
    return ((y - tt) ** 2).mean().item()
f0 = train(False); f1 = train(True)
print(f"  final: no-residual {f0:.5f}  vs  residual {f1:.5f}  "
      f"(residual is why depth-100 nets train at all)")

# TRY:
# - init_gain sweep at act=tanh: 0.5 -> norms collapse toward the input layer; 3.0 -> explode.
# - act=relu, gain=1.0 (wrong) vs sqrt(2) (He, right).
# - residual=True with tanh g=1.0 depth 24: norms stay flat from layer 1 on — the identity path
#   carries gradient around every block. (Layer 0's norm is the stem's, it can differ.)
# - Delete the `and l > 0` in forward only (keep it in backward): residual=True crashes again,
#   proving the constraint is in the MATH, not the implementation.
# - In train(): depth 16 -> 32 widens the no-residual vs residual gap.