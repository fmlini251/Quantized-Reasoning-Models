"""Shared test/bench helpers for the flash_ozaki suite.

relerr / exact_attn / do_bench / cache_mb were copy-pasted across bench_*.py and the verification
scripts; this is the single source. (The verification scripts keep their OWN production-reference
encoders/peels on purpose -- a faithfulness checker must not import the thing it checks.)
"""
import time
import torch


def relerr(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


def exact_attn(q, k, v, causal=True):
    """fp32 reference attention, general over MHA/GQA and prefill/decode.

    q:[B,Hq,T,D], k,v:[B,Hkv,N,D] (GQA: Hq=Hkv*G, the G query heads share a kv head). Query token i
    sits at abs kv position N-T+i, so causal masks qpos<kpos -- reduces to the plain tril mask when
    T==N and to "attend all" when T==1. Accepts already-replicated MHA inputs (G=1) unchanged."""
    B, Hq, T, D = q.shape
    Hkv, N = k.shape[1], k.shape[2]
    G = Hq // Hkv
    kk = (k if G == 1 else k.repeat_interleave(G, dim=1)).float()
    vv = (v if G == 1 else v.repeat_interleave(G, dim=1)).float()
    s = (q.float() @ kk.transpose(-1, -2)) / (D ** 0.5)
    if causal:
        qp = torch.arange(N - T, N, device=q.device)[:, None]
        kp = torch.arange(N, device=q.device)[None, :]
        s = s.masked_fill(qp < kp, -float("inf"))
    return torch.softmax(s, -1) @ vv


def do_bench(fn, n=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def cache_mb(kv):
    """MB of a KV-cache tuple (sum of tensor byte sizes)."""
    return sum(t.numel() * t.element_size() for t in kv if torch.is_tensor(t)) / 1e6
