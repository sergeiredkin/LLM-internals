import math, torch, torch.nn.functional as F
torch.manual_seed(0)

CFG = dict(
    B=1, H=2, T=512, d=64,
    Br=64, Bc=64,            # query-block, key-block: the ONLY structural knobs FlashAttention adds
    causal=True,
    verbose_tiny=False,      # set T=8, Br=8, Bc=4, True -> watch m/l/acc rescale block by block
    bench_sdpa=False,        # True + CUDA runtime: math vs efficient vs flash backend at T=4096
)

def naive(Q, K, V, causal):
    S = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])     # (B,H,T,T) <- the memory problem
    if causal:
        T = Q.shape[-2]
        S = S.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=Q.device), 1),
                          float("-inf"))
    return S.softmax(-1) @ V

def flash(Q, K, V, Br, Bc, causal, verbose=False):
    B, H, T, d = Q.shape
    O = torch.empty_like(Q); scale = 1 / math.sqrt(d)
    for i0 in range(0, T, Br):
        i = i0 + torch.arange(Br); i = i[i < T]              # query rows in this block
        Qi = Q[:, :, i]
        m = torch.full((B, H, len(i)), float("-inf"))        # running max per query
        l = torch.zeros(B, H, len(i))                        # running softmax denominator
        acc = torch.zeros(B, H, len(i), d)                   # running output accumulator
        for j0 in range(0, T, Bc):
            j = j0 + torch.arange(Bc)
            if causal and j[0] > i[-1]: break                # whole block in the future: skip
            j = j[j < T]
            S = Qi @ K[:, :, j].transpose(-2, -1) * scale    # (B,H,br,bc) <- ONLY block-sized mem
            if causal:
                S = S.masked_fill(j[None, :] > i[:, None], float("-inf"))
            m_new = torch.maximum(m, S.amax(-1))             # new running max
            P = torch.exp(S - m_new[..., None])              # unnormalized probs vs NEW max
            corr = torch.exp(m - m_new)                      # rescale old stats (THE trick)
            l = l * corr + P.sum(-1)
            acc = acc * corr[..., None] + P @ V[:, :, j]
            if verbose:
                print(f"  block q={i0} k={j0}: m(row0,h0) {m[0,0,0].item():.3f} -> "
                      f"{m_new[0,0,0].item():.3f}   l={l[0,0,0].item():.3f}")
            m = m_new
        O[:, :, i] = acc / l[..., None]
    return O

c = CFG
Q, K, V = (torch.randn(c["B"], c["H"], c["T"], c["d"]) for _ in range(3))
if c["verbose_tiny"]:
    out = flash(Q, K, V, c["Br"], c["Bc"], c["causal"], verbose=True)
ref = naive(Q, K, V, c["causal"])
out = flash(Q, K, V, c["Br"], c["Bc"], c["causal"])
print(f"max |flash - naive| = {(out - ref).abs().max().item():.2e}")

T, H, d = 4096, 32, 128
print(f"\npeak score memory @ T={T},H={H},d={d}:")
print(f"  naive : {1*H*T*T*4/1e9:6.2f} GB  (B,H,T,T scores, fp32; softmax needs another copy)")
print(f"  flash : {(1*H*128*128*4 + 1*H*T*d*4)/1e6:6.2f} MB  (Br*Bc block + output)")

if c["bench_sdpa"] and torch.cuda.is_available():
    q, k, v = (torch.randn(1, H, T, d, device="cuda", dtype=torch.float16) for _ in range(3))
    import time
    def timeit(fn, n=20):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n): o = fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        for name, be in [("MATH", SDPBackend.MATH), ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
                         ("FLASH", SDPBackend.FLASH_ATTENTION)]:
            try:
                with sdpa_kernel(be):
                    ms = timeit(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
                print(f"  SDPA {name:10s} {ms:8.2f} ms  peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
            except RuntimeError as e:
                print(f"  SDPA {name:10s} FAILED: {str(e)[:60]} (often OOM <- the lesson)")
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        ms = timeit(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        print(f"  SDPA auto: {ms:.2f} ms")
elif c["bench_sdpa"]:
    print("bench_sdpa needs a CUDA runtime (Colab: Runtime > Change runtime type > GPU)")

# TRY:
# - verbose_tiny=True with T=8, Br=8, Bc=4: see m jump when a new block has a bigger max, and l/acc
#   get multiplied by exp(m_old - m_new). THAT rescaling is all FlashAttention is.
# - Br=Bc=16 vs 256 on the python loop: tiny blocks = more python overhead; huge blocks = more
#   memory. Real kernels tune this per GPU (e.g. 64-128).
# - causal=True: note the early `break` — half the key blocks are never touched.
# - T=4096 on CPU with H=2: naive allocates 4096^2*2*4 = 134 MB of scores; flash allocates ~KB.
# - d=128 vs 64: scores halve/double in size but V-matmul cost doubles — attention cost is
#   O(T^2*d) for scores + O(T^2*d) for P@V.