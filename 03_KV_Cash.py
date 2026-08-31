import time, torch, torch.nn as nn
torch.manual_seed(0)

CFG = dict(
    B=1, d_model=128, n_heads=4, n_kv_heads=4,   # n_kv_heads=2 -> GQA: half the KV memory
    n_layers=2, T_prompt=64, T_gen=256,
    timeit=True,
)

class Layer(nn.Module):
    def __init__(self, d, H, Hkv):
        super().__init__()
        self.H, self.Hkv, self.dh = H, Hkv, d // H
        self.q = nn.Linear(d, H * self.dh, bias=False)
        self.k = nn.Linear(d, Hkv * self.dh, bias=False)
        self.v = nn.Linear(d, Hkv * self.dh, bias=False)
        self.o = nn.Linear(d, d, bias=False)
    def attend(self, q, k, v, mask=None):
        k = k.repeat_interleave(self.H // self.Hkv, 1)       # GQA: kv heads shared across q heads
        v = v.repeat_interleave(self.H // self.Hkv, 1)
        s = q @ k.transpose(-2, -1) * self.dh ** -0.5
        if mask is not None: s = s.masked_fill(mask, float("-inf"))
        return s.softmax(-1) @ v

class Tiny(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.layers = nn.ModuleList(
            Layer(c["d_model"], c["n_heads"], c["n_kv_heads"]) for _ in range(c["n_layers"]))
    def prompt(self, x, cnt):                                # full forward over a prefix (fills cache)
        B, T, d = x.shape
        L0 = self.layers[0]; H, Hkv, dh = L0.H, L0.Hkv, L0.dh
        i = torch.arange(T); mask = i[None, :] > i[:, None]  # future mask (T,T)
        caches, h = [], x
        for L in self.layers:
            q = L.q(h).view(B, T, H, dh).transpose(1, 2)
            k = L.k(h).view(B, T, Hkv, dh).transpose(1, 2)
            v = L.v(h).view(B, T, Hkv, dh).transpose(1, 2)
            cnt["proj_tok"] += T; cnt["score"] += T * T
            h = h + L.o(L.attend(q, k, v, mask).transpose(1, 2).reshape(B, T, -1))
            caches.append((k, v))
        return h[:, -1:], caches
    def step(self, x, caches, cnt):                          # x: (B,1,d), one token, uses cache
        B = x.shape[0]; L0 = self.layers[0]
        H, Hkv, dh = L0.H, L0.Hkv, L0.dh
        h = x
        for li, (L, (kc, vc)) in enumerate(zip(self.layers, caches)):
            q = L.q(h).view(B, 1, H, dh).transpose(1, 2)
            k = L.k(h).view(B, 1, Hkv, dh).transpose(1, 2)
            v = L.v(h).view(B, 1, Hkv, dh).transpose(1, 2)
            kc, vc = torch.cat([kc, k], 2), torch.cat([vc, v], 2)
            caches[li] = (kc, vc)
            cnt["proj_tok"] += 1; cnt["score"] += kc.shape[2]
            h = h + L.o(L.attend(q, kc, vc).transpose(1, 2).reshape(B, 1, -1))
        return h

model = Tiny(CFG).eval()
x0 = torch.randn(CFG["B"], CFG["T_prompt"], CFG["d_model"])

with torch.no_grad():
    cnt_n = {"proj_tok": 0, "score": 0}
    t0 = time.perf_counter(); gen_n = []
    xs = x0
    for _ in range(CFG["T_gen"]):                # NO cache: re-forward the whole prefix each token
        h, _ = model.prompt(xs, cnt_n); gen_n.append(h); xs = torch.cat([xs, h], 1)
    gen_n = torch.cat(gen_n, 1)
    t_nocache = time.perf_counter() - t0

    cnt_c = {"proj_tok": 0, "score": 0}
    t0 = time.perf_counter()
    h, caches = model.prompt(x0, cnt_c); gen_c = [h]
    for _ in range(CFG["T_gen"]):
        h = model.step(h, caches, cnt_c); gen_c.append(h)
    gen_c = torch.cat(gen_c, 1)
    t_cache = time.perf_counter() - t0

print(f"correctness: max|no-cache - cache| = {(gen_c[:, :CFG['T_gen']] - gen_n).abs().max().item():.2e}")
print(f"no cache : proj token-passes {cnt_n['proj_tok']:>9,}   score elems {cnt_n['score']:>12,}")
print(f"kv cache : proj token-passes {cnt_c['proj_tok']:>9,}   score elems {cnt_c['score']:>12,}")
if CFG["timeit"]:
    print(f"wall time: no-cache {t_nocache:.2f}s   cache {t_cache:.2f}s")

c = CFG; dh = c["d_model"] // c["n_heads"]; Ttot = c["T_prompt"] + c["T_gen"]
print("\nKV cache memory (2 tensors K,V, all layers):")
for name, by in [("fp32", 4), ("fp16/bf16", 2), ("int8", 1), ("int4", 0.5)]:
    mb = c["n_layers"] * 2 * c["B"] * c["n_kv_heads"] * dh * Ttot * by / 1e6
    print(f"  {name:10s} T={Ttot}: {mb:8.2f} MB")
gqa = c["n_layers"] * 2 * c["B"] * 2 * dh * Ttot * 2 / 1e6   # Hkv=2, fp16
print(f"  -> with GQA n_kv_heads=2 (fp16): {gqa:.2f} MB  (that's why Llama-2-70B uses GQA)")

# TRY:
# - n_kv_heads=2: outputs still match (GQA), cache memory halves.
# - T_gen=1024: watch score-elem counts grow ~quadratically (no-cache) vs ~linearly (cache).
# - Compute at what context length fp16 cache of a 7B model (32 layers, 32 heads, 128 dh, B=1)
#   hits 8 GB: solve 32*2*32*128*T*2 = 8e9 -> T ~ 488k... now do BATCH=32 -> T ~ 15k. That's the
#   real long-context OOM.
# - Delete the mask in prompt() during decode-only steps and see why prefill still needs it.