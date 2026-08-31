import math, torch
torch.manual_seed(3)

CFG = dict(
    temperature=0.8,     # <1 sharpen, >1 flatten, ->0 = greedy
    top_k=0,             # 0 disables; else keep k highest-prob tokens
    top_p=0.9,           # 1.0 disables; smallest set with cumulative prob >= p
    rep_penalty=1.3,     # 1.0 disables; divide positive logits of seen tokens by this
    max_words=30, greedy=False, alpha=0.5,   # alpha = count smoothing
    verbose_steps=5,
)

CORPUS = """the cat sat on the mat . the cat ate the fish . the dog sat on the log .
the dog ate the bone . a bird sat on the wire . a bird ate a seed .
the cat saw the dog . the dog saw the cat . a cat and a dog sat on the mat ."""
words = CORPUS.split()
vocab = ["<s>"] + sorted(set(words))
stoi = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
counts = torch.zeros(V, V)
seq = ["<s>"] + words
for a, b in zip(seq, seq[1:]):
    counts[stoi[a], stoi[b]] += 1

def next_logits(prev):
    return torch.log(counts[prev] + CFG["alpha"])            # unnormalized is fine

def show(p, tag, k=6):
    top = p.topk(k)
    print(f"    {tag:12s}", ", ".join(f"{vocab[i]}:{v*100:.1f}%" for v, i in zip(top.values, top.indices)))

def sample_step(logits, recent, step):
    l = logits.clone()
    if CFG["rep_penalty"] > 1.0:
        for t in set(recent):
            l[t] = l[t] / CFG["rep_penalty"] if l[t] > 0 else l[t] * CFG["rep_penalty"]
    if CFG["greedy"]:
        return l.argmax()
    probs = torch.softmax(l / CFG["temperature"], -1)
    if step < CFG["verbose_steps"]: show(probs, "after temp")
    if CFG["top_k"] > 0:
        kth = probs.topk(CFG["top_k"]).values[-1]
        probs = torch.where(probs >= kth, probs, torch.zeros_like(probs))
        if step < CFG["verbose_steps"]: show(probs, "after top-k")
    if CFG["top_p"] < 1.0:
        sp, si = probs.sort(descending=True)
        keep = (sp.cumsum(-1) - sp) < CFG["top_p"]           # keep tokens needed to reach p
        sp = sp * keep
        probs = torch.zeros_like(probs).scatter(0, si, sp)
        if step < CFG["verbose_steps"]: show(probs, "after top-p")
    probs = probs / probs.sum()
    if step < CFG["verbose_steps"]:
        nz = probs[probs > 0]
        H = -(nz * nz.log()).sum().item()
        print(f"    kept {int((probs>0).sum())}/{V} tokens, entropy {H:.2f} nats\n")
    return torch.multinomial(probs, 1).item()

print("=== generation ===")
cur, out, recent = 0, [], []
for step in range(CFG["max_words"]):
    nxt = sample_step(next_logits(cur), recent, step)
    w = vocab[nxt]
    if w == ".": out.append(w); break
    out.append(w); recent.append(nxt); cur = nxt
print(" ".join(out))

print("\n=== temperature on fixed logits [3.0, 2.5, 2.0, 1.0, 0.0, -1.0] ===")
logits = torch.tensor([3.0, 2.5, 2.0, 1.0, 0.0, -1.0])
for temp in (0.1, 0.5, 1.0, 2.0):
    p = torch.softmax(logits / temp, -1)
    H = -(p * p.clamp_min(1e-12).log()).sum().item()
    sp, _ = p.sort(descending=True)
    nuc = int(((sp.cumsum(0) - sp) < 0.9).sum())
    print(f"  T={temp:3.1f}  probs {[f'{v:.3f}' for v in p.tolist()]}  H={H:.2f}  top-p.9 keeps {nuc}")

# TRY:
# - greedy=True or temperature=0.05: infinite "the cat sat on the mat . the cat sat..." loops.
# - rep_penalty=1.0 + greedy: same; then rep_penalty=2.0: loops break but text gets weird (penalized
#   tokens are exactly the fluent ones).
# - top_p=0.5 vs 0.98 with temperature=1.5: high temp + tight top-p is the classic "creative but
#   not insane" combo; watch kept-token counts in verbose output.
# - alpha=0.05 vs 5.0: smoothing flattens the bigram distribution -> more randomness even at T=0.8.
# - Add top_a (drop tokens with prob > a) or min_p (keep p_i >= min_p * p_max) — 10 lines each,
#   same pattern. You will never fear a sampling arg again.