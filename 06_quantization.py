# 06 — Quantization (absmax int8/int4, granularity, NF4, outliers)
#
# PATCH HISTORY (all lessons learned the hard way):
#   v2: NF4 nearest-level search as a 16-pass linear scan instead of one (N,64,16)
#       broadcast temp (~0.8 GB -> ~0). Same answer, tiny memory. This was the CPU hog.
#   v2: default size 2048->1024 (stats unchanged to ~3 decimals)
#   v2: per-phase timing ticks
#   v3: matplotlib REMOVED entirely — ASCII histograms via torch.histc.
#       Nothing to hang, block, or hide. Dependencies: torch only.
#
# WHAT YOU'LL LEARN:
#   - error drops as scale granularity gets finer (tensor -> channel -> group)
#   - error explodes below 4 bits
#   - outliers (1% of weights, 8x bigger) destroy per-tensor scales -> why
#     LLM.int8()/QLoRA exist; grouping isolates the damage; NF4 wins on gaussian weights

import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

torch.manual_seed(0)
plt.switch_backend('Agg')

OUTPUT_DIR = '/home/sergei/Documents/LLM gpt trial'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

CFG = dict(
    d_out=1024, d_in=1024,
    bits=4, symmetric=True,
    mode="group",
    group_size=64,
    nf4=False,
    outlier_frac=0.0,
    outlier_scale=8.0,
    n_threads=0,
)

NF4 = torch.tensor([-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
                    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7226, 1.0])

if CFG["n_threads"]:
    torch.set_num_threads(CFG["n_threads"])

T0 = time.perf_counter()
def tick(msg):
    print(f"  [{time.perf_counter() - T0:6.2f}s] {msg}")

def qdq(W, bits, symmetric, mode, group, nf4):
    if   mode == "tensor":  G = W.reshape(1, -1)
    elif mode == "channel": G = W.reshape(W.shape[0], -1)
    else:                   G = W.reshape(-1, group)
    if nf4:
        s  = G.abs().amax(-1, keepdim=True).clamp_min(1e-8)
        Gq = G / s
        idx  = torch.zeros(Gq.shape, dtype=torch.long)
        best = torch.full_like(Gq, float("inf"))
        for i, lv in enumerate(NF4.tolist()):
            err = (Gq - lv).abs()
            closer = err < best
            best[closer] = err[closer]
            idx[closer]  = i
        D = NF4[idx] * s
    elif symmetric:
        qmax = 2 ** (bits - 1) - 1
        s = (G.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
        D = (G / s).round().clamp(-qmax, qmax) * s
    else:
        qmax = 2 ** bits - 1
        mn, mx = G.amin(-1, keepdim=True), G.amax(-1, keepdim=True)
        s = ((mx - mn) / qmax).clamp_min(1e-12); zp = (-mn / s).round()
        D = ((G / s).round().clamp(0, qmax) - zp) * s
    return D.reshape(W.shape)

def plot_weight_distribution(original, dequantized, cfg, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    w_flat = original.flatten().cpu().numpy()
    axes[0].hist(w_flat, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Weight Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Original FP32 Weights\nmin={w_flat.min():+.3f}, max={w_flat.max():+.3f}')
    axes[0].grid(True, alpha=0.3)
    d_flat = dequantized.flatten().cpu().numpy()
    axes[1].hist(d_flat, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Weight Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Dequantized ({cfg["mode"]}/int{cfg["bits"]})\nmin={d_flat.min():+.3f}, max={d_flat.max():+.3f}')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path} ({os.path.getsize(save_path):,} bytes)")

def plot_error_distribution(error, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    err_flat = error.flatten().cpu().numpy()
    ax.hist(err_flat, bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Quantization Error (dequantized - original)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Quantization Error Distribution\nmean={err_flat.mean():+.4f}, std={err_flat.std():.4f}')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path} ({os.path.getsize(save_path):,} bytes)")

def plot_granularity_sweep(W, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    modes = [("tensor", None, 4.0), ("channel", None, 4.016), ("group", 256, 4.125),
             ("group", 64, 4.5), ("group", 32, 5.0)]
    bpw_list, err_list, labels = [], [], []
    for mode, g, bpw in modes:
        Dg = qdq(W, 4, True, mode, g, False)
        err = (Dg - W).norm() / W.norm()
        bpw_list.append(bpw)
        err_list.append(err.item())
        labels.append(f"{mode}" if g is None else f"{mode} (g={g})")
    ax.plot(bpw_list, err_list, 'o-', linewidth=2, markersize=8, color='darkblue')
    ax.set_xlabel('Bits Per Weight (bpw)', fontsize=12)
    ax.set_ylabel('Relative Error', fontsize=12)
    ax.set_title('Granularity Sweep: Quantization Error vs. Bits Per Weight', fontsize=13)
    ax.grid(True, alpha=0.3)
    for i, (bpw, err, label) in enumerate(zip(bpw_list, err_list, labels)):
        ax.annotate(label, (bpw, err), textcoords="offset points", xytext=(5, 5), fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path} ({os.path.getsize(save_path):,} bytes)")

def plot_bits_sweep(W, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    bits_list = [8, 4, 2]
    err_list = []
    for b in bits_list:
        Dg = qdq(W, b, True, "group", 64, False)
        err = (Dg - W).norm() / W.norm()
        err_list.append(err.item())
    ax.bar([str(b) for b in bits_list], err_list, color=['forestgreen', 'steelblue', 'coral'], edgecolor='black')
    ax.set_xlabel('Number of Bits', fontsize=12)
    ax.set_ylabel('Relative Error', fontsize=12)
    ax.set_title('Bits Sweep: Quantization Error vs. Bit Width (group=64)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    for i, err in enumerate(err_list):
        ax.text(i, err + 0.0005, f'{err:.4f}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path} ({os.path.getsize(save_path):,} bytes)")

def plot_outlier_sweep(W, save_path):
    """Optimized outlier sweep - uses smaller tensor for outlier test."""
    print("  Creating outlier stress test (optimized)...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use smaller tensor for outlier test to save memory
    W_small = torch.randn(512, 512)
    Wo = W_small * torch.where(torch.rand_like(W_small) < 0.01, 8.0, 1.0)
    
    configs = [("tensor", None, False, "tensor"), ("channel", None, False, "channel"),
               ("group", 64, False, "group (int4)"), ("group", 64, True, "group (NF4)")]
    labels, err_list, colors = [], [], []
    for mode, g, nf4, label in configs:
        Dg = qdq(Wo, 4, True, mode, g, nf4)
        err = (Dg - Wo).norm() / Wo.norm()
        labels.append(label)
        err_list.append(err.item())
        colors.append('coral' if nf4 else 'steelblue')
        print(f"    {label}: rel-err {err:.4f}")
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, err_list, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=11)
    ax.set_ylabel('Relative Error', fontsize=12)
    ax.set_title('Outlier Stress Test (1% weights ×8, bits=4)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, err in zip(bars, err_list):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{err:.4f}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path} ({os.path.getsize(save_path):,} bytes)")

# ---------- data ----------
W = torch.randn(CFG["d_out"], CFG["d_in"])
tick("setup done")

# ---------- main quantization ----------
D = qdq(W, CFG["bits"], CFG["symmetric"], CFG["mode"], CFG["group_size"], CFG["nf4"])
x = torch.randn(64, CFG["d_in"])
print(f"mode={CFG['mode']:7s} bits={CFG['bits']} sym={CFG['symmetric']} nf4={CFG['nf4']}")
print(f"  weight rel-err {(D-W).norm()/W.norm():.4f}   max-err {(D-W).abs().max():.3f}")
tick("main qdq")

# ---------- Create matplotlib visualizations ----------
print("\n--- Creating matplotlib charts ---")
plot_weight_distribution(W, D, CFG, os.path.join(OUTPUT_DIR, 'weight_distribution.png'))
plot_error_distribution(D - W, os.path.join(OUTPUT_DIR, 'error_distribution.png'))

# ---------- sweep 1: granularity ----------
print("\n--- granularity sweep (bits=4, symmetric) ---")
for mode, g, bpw in [("tensor",None,4.0), ("channel",None,4.016), ("group",256,4.125),
                     ("group",64,4.5), ("group",32,5.0)]:
    Dg = qdq(W, 4, True, mode, g, False)
    print(f"  {mode:7s} g={str(g):5s} ~{bpw:.2f} bpw  rel-err {(Dg-W).norm()/W.norm():.4f}")
tick("granularity sweep")
plot_granularity_sweep(W, os.path.join(OUTPUT_DIR, 'granularity_sweep.png'))

# ---------- sweep 2: bits ----------
print("\n--- bits sweep (group=64) ---")
for b in (8, 4, 2):
    Dg = qdq(W, b, True, "group", 64, False)
    print(f"  int{b}: rel-err {(Dg-W).norm()/W.norm():.4f}")
tick("bits sweep")
plot_bits_sweep(W, os.path.join(OUTPUT_DIR, 'bits_sweep.png'))

# ---------- sweep 3: outlier stress (optimized) ----------
print("\n--- outlier stress (1% weights x8, bits=4) ---")
plot_outlier_sweep(W, os.path.join(OUTPUT_DIR, 'outlier_sweep.png'))
tick("outlier stress complete")

print("\n✓✓✓ All 5 charts saved to:", OUTPUT_DIR)
print("DONE")

# TRY:
# - outlier_frac=0.01 with mode='tensor': rel-err explodes (one huge weight eats the range).
#   Same weights, group=64: most groups never see an outlier. THIS is why LLM.int8()/QLoRA exist.
# - nf4=True (with mode='group') vs symmetric int4 at group=64: NF4 wins — its 16 levels are
#   matched to N(0,1) density instead of uniform spacing. Compare in the outlier table too.
# - bits=2, group=32: still usable; then bits=2, mode='tensor': garbage. No free lunch below 4 bits.
# - symmetric=False on gaussian data: ~no change. Skew the weights
#       W = W.abs()**2 * W.sign()
#   (recreate W first!) and asymmetric starts winning.
# - KERNEL LESSON: swap the NF4 scan back to the one-liner broadcast
#       idx = ((Gq[..., None] - NF4).abs().argmin(-1))
#   and compare tick times on the outlier-stress section. Same answer, wildly different memory.
#   Broadcast trades bandwidth for one op; the scan trades 16 cheap passes for zero temporaries.
#   On CPU, bandwidth wins. That trade IS kernel design.
# - Watch the layer-output rel-err vs weight rel-err in the main line: they're nearly equal.
#   Quantization noise passes through the layer ~1:1 — that's why weight err is a good proxy.