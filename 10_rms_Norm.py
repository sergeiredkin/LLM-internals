# 10 — RMSNorm (what it normalizes, what it refuses to, why LLMs picked it)
# PATCHED v2:
#   (1) section 5: manual dL/dg must sum over ALL non-channel dims — was .sum(0) on a
#       (4,8,d) tensor -> broadcast garbage -> printed 2.16e+01 instead of ~1e-5.
#   (2) section 6: F.layer_norm(input, normalized_shape, weight) — 2nd positional arg is a
#       TUPLE OF INTS, not the gain. bench(F.layer_norm) crashed; now an adapter lambda.
#   (3) bench sized down for CPU comfort.

import time, torch, torch.nn.functional as F
torch.manual_seed(0)

CFG = dict(d=512, eps=1e-6)

def rms_norm(x, g=None, eps=1e-6):
    xhat = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return xhat if g is None else xhat * g

d = CFG["d"]; eps = CFG["eps"]
x = torch.randn(4, 8, d)
g = torch.ones(d)

print("--- 1. what it guarantees: per-token RMS = 1 ---")
xh = rms_norm(x, g, eps)
print("  RMS per token:", [f"{v:.4f}" for v in xh.pow(2).mean(-1).sqrt().flatten()[:6].tolist()])
print("  (it does NOT guarantee zero mean -> next test)")

print("\n--- 2. scale invariance (both norms) ---")
for name, fn in [("RMS", lambda z: rms_norm(z, g, eps)),
                 ("LN ", lambda z: F.layer_norm(z, (d,), g))]:
    diff = (fn(x * 1000) - fn(x)).abs().max().item()
    print(f"  {name}: ||norm(1000x) - norm(x)||_max = {diff:.2e}")
print("  both ~0 -> scale-invariant. Note LN's residual is ~10x bigger: mean-subtraction")
print("  amplifies float cancellation. Scale-invariance is numerically sturdier.")

print("\n--- 3. SHIFT: the actual difference ---")
for name, fn in [("RMS", lambda z: rms_norm(z, g, eps)),
                 ("LN ", lambda z: F.layer_norm(z, (d,), g))]:
    diff = (fn(x + 5) - fn(x)).abs().max().item()
    mean_after = fn(x + 5).mean(-1).abs().mean().item()
    print(f"  {name}: ||norm(x+5) - norm(x)||_max = {diff:.3f}   mean of output {mean_after:.3f}")
print("  LN removes the mean; RMS doesn't. Transformer activations have no useful mean")
print("  structure to remove -> LN's mean-subtraction + bias are wasted compute. Hence RMSNorm.")

print("\n--- 4. eps and fp16 underflow ---")
x16 = (torch.randn(4, 8, d) * 1e-4).half()
for e in (0.0, 1e-3):
    y = rms_norm(x16, None, e)
    bad = torch.isinf(y).any().item() or torch.isnan(y).any().item()
    print(f"  fp16 x~1e-4, eps={e:<6}: output has inf/nan? {bad}   "
          f"(x^2 ~ 1e-8 underflows fp16; eps keeps rsqrt finite)")
print("  production kernels also upcast the reduction to fp32.")

print("\n--- 5. gradients (via autograd) ---")
xa = x.clone().requires_grad_(True); ga = torch.randn(d, dtype=x.dtype, requires_grad=True)
R = torch.randn_like(x)
rms_norm(xa, ga, eps).mul(R).sum().backward()
xhat = rms_norm(x, None, eps)
manual_dg = (R * xhat).sum((0, 1))                    # PATCHED: sum over dims 0 AND 1
print(f"  dL/dg vs manual sum(R*xhat): {(ga.grad - manual_dg).abs().max().item():.2e}")
xl = x.clone().requires_grad_(True); xr = x.clone().requires_grad_(True)
F.layer_norm(xl, (d,), torch.ones(d)).mul(R).sum().backward()
rms_norm(xr, None, eps).mul(R).sum().backward()
print(f"  LN dL/dx vs RMS dL/dx max diff: {(xl.grad - xr.grad).abs().max().item():.3f} "
      f"(LN's dx carries extra mean/var coupling terms)")

print("\n--- 6. microbenchmark fwd+bwd ---")
xb = torch.randn(1, 2048, 4096, requires_grad=True)   # PATCHED: smaller, CPU-friendly
gb = torch.ones(4096, requires_grad=True)
def bench(fn, n=10):
    t0 = time.perf_counter()
    for _ in range(n):
        y = fn(xb, gb); y.sum().backward()
        xb.grad = None; gb.grad = None
    return (time.perf_counter() - t0) / n * 1e3
ln_fn = lambda z, w: F.layer_norm(z, (z.shape[-1],), w)  # PATCHED: adapter — normalized_shape
print(f"  RMSNorm {bench(rms_norm):7.1f} ms/iter    LayerNorm {bench(ln_fn):7.1f} ms/iter")
print(f"  params per channel: RMS 1 (g) vs LN 2 (g,b)")

print("\nDONE")

# TRY:
# - eps=1e12: everything squashes toward g (rsqrt(x^2+eps) -> 0). eps is not decoration.
# - Feed a constant vector (all d values equal): LN output ~0 (mean removed, eps saves rsqrt),
#   RMS output = g * sign. A failure mode LN has that RMS doesn't.
# - Pre-norm vs post-norm: rms_norm(x) @ W vs rms_norm(x @ W) — run script 09's gradient-norm
#   plot for both; pre-norm's identity path is the residual trick again.
# - Implement RMSNorm's dL/dx by hand and check vs autograd, like script 09.

# TRY:
# - eps=1e12: everything squashes toward g (rsqrt(x^2+eps) -> 0). eps is not decoration.
# - Feed a constant vector (all d values equal): LN output = 0 (mean removed, std 0 -> eps saves
#   it), RMS output = g * sign. A whole failure mode LN has and RMS doesn't.
# - Pre-norm vs post-norm: wrap a linear layer with rms_norm(x) @ W (pre) vs rms_norm(x @ W)
#   (post), run script-09's gradient-norm plot with both — pre-norm's identity path is the same
#   trick as residuals.
# - Implement RMSNorm backward by hand (dL/dx = ...) and check against autograd like script 09.