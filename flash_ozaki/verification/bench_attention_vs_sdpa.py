"""Full-attention comparison: flash-ozaki1_fp (codegen, production-faithful) vs torch SDPA and vs the
same kernel's EXACT bf16 flash path (ozaki off). Accuracy (vs fp32 exact attention) + latency, for
prefill (MHA) and decode (GQA head-fold, Qwen-7B shape). Run from repo root:
    CUDA_VISIBLE_DEVICES=2 python flash_ozaki/verification/bench_attention_vs_sdpa.py"""
import os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, "/home/howonlee/Quantized-Reasoning-Models")
from flash_ozaki.flash_oz1fp_codegen import flash_oz1fp_cg
from flash_ozaki._testutil import relerr, exact_attn, do_bench
dev = "cuda"; torch.manual_seed(0)


def repeat_kv(x, G): return x if G == 1 else x.repeat_interleave(G, dim=1)


def sdpa(q, k, v):
    G = q.shape[1] // k.shape[1]
    return F.scaled_dot_product_attention(q, repeat_kv(k, G), repeat_kv(v, G), is_causal=(q.shape[2] > 1))


D = 128; W, NMP = 4, 10                                      # ozaki config for the attention emulation
print(f"=== Accuracy vs fp32-exact attention (ozaki w{W} nmp{NMP}) ===")
print(f"  {'case':>22} | {'flash-ozaki vs exact':>20} | {'flash-exact vs exact':>20} | {'SDPA vs exact':>13}")
# prefill MHA
B, H, N = 1, 28, 1024
q = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16); k = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16); v = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16)
ex = exact_attn(q, k, v)
oz = flash_oz1fp_cg(q, k, v, nmp=NMP, w=W, causal=True, BLOCK_M=32, BLOCK_N=32)
fe = flash_oz1fp_cg(q, k, v, nmp=1, w=8, causal=True, ozaki=False)
print(f"  {'PREFILL MHA N=1024':>22} | {relerr(oz,ex):20.2e} | {relerr(fe,ex):20.2e} | {relerr(sdpa(q,k,v),ex):13.2e}")
# decode GQA
Bd, Hq, Hkv, Nd = 32, 28, 4, 2048
qd = torch.randn(Bd, Hq, 1, D, device=dev, dtype=torch.bfloat16); kd = torch.randn(Bd, Hkv, Nd, D, device=dev, dtype=torch.bfloat16); vd = torch.randn(Bd, Hkv, Nd, D, device=dev, dtype=torch.bfloat16)
exd = exact_attn(qd, kd, vd)
ozd = flash_oz1fp_cg(qd, kd, vd, nmp=NMP, w=W, causal=True)
fed = flash_oz1fp_cg(qd, kd, vd, nmp=1, w=8, causal=True, ozaki=False)
print(f"  {'DECODE GQA B=32 N=2048':>22} | {relerr(ozd,exd):20.2e} | {relerr(fed,exd):20.2e} | {relerr(sdpa(qd,kd,vd),exd):13.2e}")

print(f"\n=== Latency ms/call ===")
print(f"  {'case':>26} | {'flash-ozaki':>11} | {'flash-exact':>11} | {'torch SDPA':>10} | {'oz/sdpa':>7}")
for N in [1024, 2048, 4096]:
    q = torch.randn(1, 28, N, D, device=dev, dtype=torch.bfloat16); k = torch.randn(1, 28, N, D, device=dev, dtype=torch.bfloat16); v = torch.randn(1, 28, N, D, device=dev, dtype=torch.bfloat16)
    t_oz = do_bench(lambda: flash_oz1fp_cg(q, k, v, nmp=NMP, w=W, causal=True, BLOCK_M=32, BLOCK_N=32))
    t_fe = do_bench(lambda: flash_oz1fp_cg(q, k, v, nmp=1, w=8, causal=True, ozaki=False))
    t_sd = do_bench(lambda: sdpa(q, k, v))
    print(f"  {('PREFILL MHA N='+str(N)):>26} | {t_oz:11.3f} | {t_fe:11.3f} | {t_sd:10.3f} | {t_oz/t_sd:6.1f}x")
for Bd in [8, 32]:
    qd = torch.randn(Bd, 28, 1, D, device=dev, dtype=torch.bfloat16); kd = torch.randn(Bd, 4, 2048, D, device=dev, dtype=torch.bfloat16); vd = torch.randn(Bd, 4, 2048, D, device=dev, dtype=torch.bfloat16)
    t_oz = do_bench(lambda: flash_oz1fp_cg(qd, kd, vd, nmp=NMP, w=W, causal=True))
    t_fe = do_bench(lambda: flash_oz1fp_cg(qd, kd, vd, nmp=1, w=8, causal=True, ozaki=False))
    t_sd = do_bench(lambda: sdpa(qd, kd, vd))
    print(f"  {('DECODE GQA B='+str(Bd)+' N=2048'):>26} | {t_oz:11.3f} | {t_fe:11.3f} | {t_sd:10.3f} | {t_oz/t_sd:6.1f}x")
print("\nDONE")
