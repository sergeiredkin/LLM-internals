import math, torch, torch.nn as nn
torch.manual_seed(0)

CFG = dict(
    d_in=512, d_out=512,
    r=16, alpha=32, lora_dropout=0.0,
    target_rank=8,               # rank of the "finetuning delta" the task needs. Set > r to break LoRA
    steps=600, bs=64, lr_full=1e-3, lr_lora=1e-2,
    qlora=False, group_size=64,  # True: freeze an NF4-quantized base (QLoRA style)
)

NF4 = torch.tensor([-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
                    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7226, 1.0])  # QLoRA paper

def nf4_dequantize(W, group):
    G = W.reshape(-1, group)
    s = G.abs().amax(1, keepdim=True).clamp_min(1e-8)
    idx = ((G / s)[..., None] - NF4).abs().argmin(-1)        # nearest NF4 level per weight
    dq = (NF4[idx] * s).reshape(W.shape)
    return dq, ((dq - W).norm() / W.norm()).item()

class LoRALinear(nn.Module):
    def __init__(self, W_base, r, alpha, p_drop=0.0):
        super().__init__()
        d_out, d_in = W_base.shape
        self.register_buffer("W", W_base)                     # frozen, serialized, follows .to()
        self.A = nn.Parameter(torch.randn(r, d_in) / math.sqrt(d_in))  # gaussian
        self.B = nn.Parameter(torch.zeros(d_out, r))                   # ZEROS -> delta starts at 0
        self.scale = alpha / r
        self.drop = nn.Dropout(p_drop)
    def forward(self, x):
        return x @ self.W.T + self.scale * (self.drop(x) @ self.A.T @ self.B.T)
    def delta(self):
        return self.scale * self.B @ self.A                   # (d_out, d_in), rank <= r

def make_problem(c):
    W0 = torch.randn(c["d_out"], c["d_in"]) / math.sqrt(c["d_in"])     # "pretrained"
    U = torch.randn(c["d_out"], c["target_rank"]) / c["target_rank"] ** 0.5
    Vt = torch.randn(c["target_rank"], c["d_in"]) / c["target_rank"] ** 0.5
    W_target = W0 + 2.0 * U @ Vt                                        # low-rank finetune delta
    x = torch.randn(4096, c["d_in"]); y = x @ W_target.T
    return W0, W_target, x, y

def train(mode, c, problem):
    W0, W_target, x, y = problem
    base_mse = ((x @ W0.T - y) ** 2).mean().item()
    if mode == "full":
        W = W0.clone().requires_grad_(True)
        fwd = lambda z: z @ W.T; params = [W]; lora = None
    else:
        base = W0
        if c["qlora"]:
            base, qerr = nf4_dequantize(W0, c["group_size"])
            print(f"  NF4 base rel-err: {qerr:.4f}   (base stays FROZEN)")
        lora = LoRALinear(base, c["r"], c["alpha"], c["lora_dropout"])
        fwd = lora; params = [lora.A, lora.B]
    opt = torch.optim.AdamW(params, lr=c["lr_full"] if mode == "full" else c["lr_lora"])
    # Every mode sees the same deterministic minibatch/noise sequence.
    data_rng = torch.Generator(device=W0.device).manual_seed(1)
    for s in range(c["steps"]):
        xb = torch.randn(c["bs"], c["d_in"], generator=data_rng, device=W0.device)
        noise = torch.randn(c["bs"], c["d_out"], generator=data_rng, device=W0.device)
        yb = xb @ W_target.T + 0.01 * noise
        loss = ((fwd(xb) - yb) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 200 == 0: print(f"  step {s:4d}  mse {loss.item():.5f}")
    final = ((fwd(x) - y) ** 2).mean().item()
    if lora is not None:
        sv = torch.linalg.svdvals(lora.delta())
        print(f"  rank of learned dW: {(sv > 1e-5).sum().item()} (<= r={c['r']}); "
              f"svals[:3] {sv[:3].tolist()} svals[r:r+2] {sv[c['r']:c['r']+2].tolist()}")
    return final, base_mse

problem = make_problem(CFG)
W0, _, x_eval, y_eval = problem
base_mse = ((x_eval @ W0.T - y_eval) ** 2).mean().item()
print(f"frozen base MSE (no training): {base_mse:.4f}")
for mode, tag in [("full", "full FT"), ("lora", "LoRA"), ("lora", "QLoRA")]:
    cfg = {**CFG, "qlora": tag == "QLoRA"}
    print(f"{tag:10s}:", ); final, _ = train(mode, cfg, problem)
    print(f"{tag:10s} FINAL MSE {final:.5f}\n")

shapes = [("q",4096,4096),("k",4096,4096),("v",4096,4096),("o",4096,4096),
          ("gate",4096,11008),("up",4096,11008),("down",11008,4096)]     # Llama-2-7B block
full = sum(o * i for _, o, i in shapes)
print(f"one 7B transformer block: full trainable {full/1e6:.0f}M")
for r in (8, 16, 64):
    l = sum(r * (i + o) for _, o, i in shapes)
    print(f"  LoRA r={r:3d}: {l/1e6:5.1f}M trainable ({100*l/full:.2f}%)")

# TRY:
# - target_rank=64 with r=16: LoRA's MSE plateaus ABOVE full FT — the update needs more rank than
#   you gave it. This is the entire bet of LoRA, made visible.
# - alpha=16 vs 32 (scale 1 vs 2): same direction, 2x effective step. Set alpha=r and double
#   lr_lora — nearly identical curve. (Why papers say "alpha/r is part of the LR".)
# - lora_dropout=0.3: adapter-only dropout; base path untouched.
# - group_size=32 vs 256 in QLoRA: watch NF4 rel-err and final MSE move.
# - Init B randomly instead of zeros -> training starts by destroying the base output. That's why
#   B=0 is not a style choice, it's the only safe init.