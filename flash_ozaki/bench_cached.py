"""Measure the kv_cache=True speedup for Flash-Ozaki1_fp (GPU 0). The cache (pre-encoded K/V
digit planes + scales) is built ONCE (amortized, like weight_cache); we time only the flash kernel."""
import sys, time
import torch
sys.path.insert(0, "/home/howonlee/Quantized-Reasoning-Models")
from flash_ozaki.flash_oz1fp_triton import flash_oz1fp
from flash_ozaki.flash_oz1fp_cached import flash_oz1fp_cached, encode_kv
from flash_ozaki.oz1fp_triton import oz1fp_params


def relerr(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


def exact_attn(q, k, v, causal=True):
    B, H, N, D = q.shape
    s = (q.float() @ k.float().transpose(-1, -2)) / (D ** 0.5)
    if causal:
        m = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
        s = s.masked_fill(~m, -float("inf"))
    return torch.softmax(s, -1) @ v.float()


def do_bench(fn, n=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


dev = "cuda"; torch.manual_seed(0)
B, H, D, causal = 1, 28, 128, True

print("=== Accuracy: cached vs non-cached vs EXACT (N=1024) ===")
N = 1024
q = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16)
ex = exact_attn(q, k, v, causal)
for (w, nmp) in [(8, 1), (4, 10)]:
    nc = flash_oz1fp(q, k, v, nmp=nmp, w=w, causal=causal, ozaki=True)
    kv = encode_kv(k, v, nmp, w)
    cc = flash_oz1fp_cached(q, kv, nmp=nmp, w=w, causal=causal)
    print(f"  w={w} nmp={nmp:<2}: cached vs EXACT={relerr(cc,ex):.2e}  non-cached vs EXACT={relerr(nc,ex):.2e}  cached vs non-cached={relerr(cc,nc):.2e}")

for (w, nmp) in [(8, 1), (4, 10)]:
    print(f"\n=== Speed w={w} nmp={nmp} (nD={oz1fp_params(nmp,w)[0]}); ms/call; cache pre-built ===")
    print(f"{'N_ctx':>6} | {'cached (kv_cache=T)':>19} | {'non-cached':>11} | {'exact flash':>11} | {'cache/noncache':>14} | {'cache/exact':>11}")
    for N in [1024, 2048, 4096]:
        q = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16)
        kv = encode_kv(k, v, nmp, w)                          # one-time cache build (not timed)
        t_cc = do_bench(lambda: flash_oz1fp_cached(q, kv, nmp=nmp, w=w, causal=causal))
        t_nc = do_bench(lambda: flash_oz1fp(q, k, v, nmp=nmp, w=w, causal=causal, ozaki=True))
        t_ex = do_bench(lambda: flash_oz1fp(q, k, v, nmp=1, w=8, causal=causal, ozaki=False))
        print(f"{N:6} | {t_cc:19.2f} | {t_nc:11.2f} | {t_ex:11.2f} | {t_nc/t_cc:13.2f}x | {t_cc/t_ex:10.2f}x")
