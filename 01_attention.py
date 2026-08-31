import torch, math
torch.manual_seed(0)

CFG = dict(
    B=2, H=4, T=6, d_model=16,   # H must divide d_model
    cross_attn=False,            # True: Q from 4 decoder tokens, K/V from 6 encoder tokens
    causal=True,
    scale_by_sqrt_d=True,
    mask_bug=False,              # True: mask flipped -> tokens attend to the FUTURE (silent!)
    skip_head_split=False,       # True: classic silent bug - attention without splitting heads
)

def split_heads(x, H):
    B, T, D = x.shape
    return x.view(B, T, H, D // H).transpose(1, 2)          # (B,T,D) -> (B,H,T,dk)

def merge_heads(x):
    B, H, T, dk = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, H * dk)

def attention(Q, K, V, cfg, label=""):
    Tq, Tkv = Q.shape[2], K.shape[2]
    scale = 1 / math.sqrt(Q.shape[-1]) if cfg["scale_by_sqrt_d"] else 1.0
    print(f"[{label}] Q {tuple(Q.shape)} K {tuple(K.shape)} V {tuple(V.shape)}")
    scores = Q @ K.transpose(-2, -1) * scale
    print(f"[{label}] scores {tuple(scores.shape)}   <- always (B,H,Tq,Tkv); THIS is what OOMs")
    if cfg["causal"]:
        allow = torch.tril(torch.ones(Tq, Tkv, dtype=torch.bool))     # broadcasts over B,H
        if cfg["mask_bug"]:
            allow = torch.triu(torch.ones(Tq, Tkv, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(~allow, float("-inf"))
    attn = scores.softmax(-1)
    print(f"[{label}] attn  {tuple(attn.shape)}  row0 weights: "
          f"{[round(w,2) for w in attn[0,0,0].tolist()]}  rowsums={attn.sum(-1)[0,0,0].item():.3f}")
    out = attn @ V
    print(f"[{label}] out   {tuple(out.shape)} -> merged {tuple(merge_heads(out).shape)}\n")
    return merge_heads(out)

B, H, T, D = CFG["B"], CFG["H"], CFG["T"], CFG["d_model"]
x_dec = torch.randn(B, 4 if CFG["cross_attn"] else T, D)
x_enc = torch.randn(B, T, D)
Wq, Wk, Wv = (torch.randn(D, D) / D**0.5 for _ in range(3))
Q, K, V = split_heads(x_dec @ Wq, H), split_heads(x_enc @ Wk, H), split_heads(x_enc @ Wv, H)
attention(Q, K, V, CFG, label="self" if not CFG["cross_attn"] else "cross")

if CFG["skip_head_split"]:
    S = (x_dec @ Wq) @ (x_enc @ Wk).transpose(-2, -1) / math.sqrt(D)
    print("no-head-split:", tuple(S.shape), "<- RUNS FINE. mixes dk into keys. silently wrong.")

if CFG["mask_bug"]:
    attention(Q, K, V, CFG, label="BUG")
    print("note: rowsums are still 1.0 -> 'softmax sums to 1' proves NOTHING about the mask.")

print("\n--- OOM math (fp32) ---")
for T2 in (512, 2048, 4096, 16384):
    el = 1 * 32 * T2 * T2     # B=1, H=32
    print(f"T={T2:6d}: score tensor = {el/1e9:.2f}G elems = {el*4/1e9:.2f} GB "
          f"(softmax + backward keep more copies)")

# TRY:
# - H=5 (doesn't divide 16) -> view() RuntimeError. Read the error.
# - cross_attn=True: confirm scores become (B,H,4,6); causal mask still works (rectangular tril).
# - mask_bug=True: row0 of attn puts mass on FUTURE keys; loss would still train. This bug is why
#   people unit-test masks.
# - scale_by_sqrt_d=False: logits get large -> softmax saturates -> attn ~ one-hot. This is WHY
#   the 1/sqrt(dk) exists.
# - Set T=6, H=1, then H=4 and diff the merged outputs: heads really do see different subspaces.