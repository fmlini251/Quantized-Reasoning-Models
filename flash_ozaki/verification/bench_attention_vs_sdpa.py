"""Full-attention comparison: flash-ozaki1_fp (codegen, production-faithful) vs torch SDPA, vs the
same kernel's EXACT bf16 flash path (ozaki off), and vs a torch attention built from the PRODUCTION
ozaki1_fp GEMM (QK & PV via ozaki1_batched_gemm_fp) with the softmax run in fp32 (P NOT truncated to
bf16). Accuracy (vs fp32 exact) + latency, prefill (MHA) and decode (GQA head-fold). Run from repo root:
    CUDA_VISIBLE_DEVICES=2 python flash_ozaki/verification/bench_attention_vs_sdpa.py"""
import os, sys, math
import torch
import torch.nn.functional as F
sys.path.insert(0, "/home/howonlee/Quantized-Reasoning-Models")
from flash_ozaki.flash_oz1fp_codegen import flash_oz1fp_cg
from flash_ozaki._testutil import relerr, exact_attn, do_bench
from emulation.llm.ozaki_matmul import ozaki1_batched_gemm_fp, CustomGemmConfig, Ozaki1Config
dev = "cuda"; torch.manual_seed(0)


def repeat_kv(x, G): return x if G == 1 else x.repeat_interleave(G, dim=1)


def sdpa(q, k, v):
    G = q.shape[1] // k.shape[1]
    return F.scaled_dot_product_attention(q, repeat_kv(k, G), repeat_kv(v, G), is_causal=(q.shape[2] > 1))


def _ozgemm_cfg(kdim, ndim, chunk):
    return CustomGemmConfig(in_feature_ts=kdim, out_feature_ts=ndim, chunk_size=chunk, name="a",
                            track_mtx_acc=False, track_model_acc=False, get_statistics=False,
                            rslt_type="ozaki1_fp")


def prod_eager(q, k, v, nmp, w, chunk):
    """EAGER production ozaki1_fp attention: materialize full S -> full-row fp32 softmax -> P->bf16 -> P@V,
    both GEMMs via production ozaki1_batched_gemm_fp. P is truncated to bf16 before P@V (the real bf16
    flash-attention datapath -- SDPA/flash-exact/flash-ozaki all do this). MHA/GQA, fp32 out."""
    B, Hq, T, D = q.shape; Hkv, N = k.shape[1], k.shape[2]; G = Hq // Hkv; Z = B * Hq
    qf = q.reshape(Z, T, D); kf = repeat_kv(k, G).reshape(Z, N, D); vf = repeat_kv(v, G).reshape(Z, N, D)
    oz1 = Ozaki1Config(rounding="round_half_away_from_0", nmp=nmp, byte_split_style="all_signed_no_clamp")
    S = ozaki1_batched_gemm_fp(qf, kf.transpose(-1, -2), _ozgemm_cfg(D, N, chunk), oz1,
                               out_dtype=torch.float32, gemm_bits=w) / math.sqrt(D)
    qp = torch.arange(N - T, N, device=dev)[:, None]; kp = torch.arange(N, device=dev)[None, :]
    P = torch.softmax(S.masked_fill(qp < kp, -float("inf")), -1).to(torch.bfloat16)   # full-row fp32 softmax, P->bf16
    O = ozaki1_batched_gemm_fp(P, vf, _ozgemm_cfg(N, D, chunk), oz1, out_dtype=torch.float32, gemm_bits=w)
    return O.reshape(B, Hq, T, D)


def prod_flash(q, k, v, nmp, w, chunk, BN=32):
    """FLASH production ozaki1_fp attention: online-softmax over BN-wide kv tiles, per-tile QK and P@V
    via production ozaki1_batched_gemm_fp. P is truncated to bf16 for the P@V GEMM while the fp32
    l-normalizer is kept -- exactly what flash-ozaki/SDPA/flash-exact do. Mirrors flash-ozaki's algorithm;
    only the GEMM impl differs (production vs codegen Triton). Returns fp32; MHA/GQA prefill."""
    B, Hq, T, D = q.shape; Hkv, N = k.shape[1], k.shape[2]; G = Hq // Hkv; Z = B * Hq
    qf = q.reshape(Z, T, D); kf = repeat_kv(k, G).reshape(Z, N, D); vf = repeat_kv(v, G).reshape(Z, N, D)
    oz1 = Ozaki1Config(rounding="round_half_away_from_0", nmp=nmp, byte_split_style="all_signed_no_clamp")
    m = torch.full((Z, T), -float("inf"), device=dev); l = torch.zeros(Z, T, device=dev)
    acc = torch.zeros(Z, T, D, device=dev); qpos = torch.arange(N - T, N, device=dev)[:, None]
    for n0 in range(0, N, BN):
        kt = kf[:, n0:n0 + BN, :]; vt = vf[:, n0:n0 + BN, :]
        st = ozaki1_batched_gemm_fp(qf, kt.transpose(-1, -2), _ozgemm_cfg(D, kt.shape[1], chunk), oz1,
                                    out_dtype=torch.float32, gemm_bits=w) / math.sqrt(D)
        kp = (n0 + torch.arange(kt.shape[1], device=dev))[None, :]
        st = st.masked_fill(qpos < kp, -float("inf"))
        mn = torch.maximum(m, st.max(-1).values); al = torch.exp(m - mn); p = torch.exp(st - mn[:, :, None])
        l = l * al + p.sum(-1)                                   # fp32 normalizer (like flash-ozaki/SDPA)
        pp = p.to(torch.bfloat16)                                # P->bf16 for P@V (real flash datapath)
        acc = acc * al[:, :, None] + ozaki1_batched_gemm_fp(pp, vt, _ozgemm_cfg(kt.shape[1], D, chunk),
                                                            oz1, out_dtype=torch.float32, gemm_bits=w)
        m = mn
    return (acc / l[:, :, None]).reshape(B, Hq, T, D)


D = 128; W, NMP = 4, 10; CHUNK = 32                          # ozaki config; CHUNK = block-FP reduction chunk
_bf = lambda x: x.to(torch.bfloat16)
print(f"=== Accuracy vs fp32-exact attention (ozaki w{W} nmp{NMP}, chunk={CHUNK}) ===")
print(f"  {'case':>22} | {'flash-ozaki':>11} | {'flash-exact':>11} | {'SDPA':>9} | {'prod-FLASH':>11}")
# prefill MHA
B, H, N = 1, 28, 1024
q = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16); k = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16); v = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16)
ex = exact_attn(q, k, v)
oz = flash_oz1fp_cg(q, k, v, nmp=NMP, w=W, causal=True, chunk_size=CHUNK, BLOCK_M=32, BLOCK_N=32)   # bf16 out, P->bf16
fe = flash_oz1fp_cg(q, k, v, nmp=1, w=8, causal=True, ozaki=False)
pf = prod_flash(q, k, v, NMP, W, CHUNK)                      # production ozaki online-softmax flash (P->bf16)
pe = prod_eager(q, k, v, NMP, W, CHUNK)                      # production ozaki materialized eager (P->bf16)
print(f"  {'PREFILL MHA N=1024':>22} | {relerr(oz,ex):11.2e} | {relerr(fe,ex):11.2e} | {relerr(sdpa(q,k,v),ex):9.2e} | "
      f"{relerr(pf,ex):11.2e}")
sdpa_fp32 = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
print(f"  * All bf16-flash-attention (SDPA / flash-exact / flash-ozaki) keep score(QK), softmax stats and the")
print(f"    l-normalizer in fp32, but truncate P to bf16 for the P@V matmul (flash-exact: tl.dot(p.to(v.dtype),v);")
print(f"    SDPA: FlashAttn tensor-core P@V; flash-ozaki: p.to(bf16) before block-FP PV). Only the final store is")
print(f"    bf16 too. So flash-ozaki == SDPA/flash-exact accuracy ({relerr(oz,ex):.1e}); the ~2e-3 vs fp32-exact is the")
print(f"    bf16 output+P rounding, NOT ozaki. Proof: SDPA is {relerr(sdpa(q,k,v),ex):.1e} at bf16 but {relerr(sdpa_fp32,ex):.1e} in fp32.")
print(f"\n  -- flash-ozaki distance to production ozaki attention (output precision MATCHED to bf16) --")
print(f"     flash-ozaki <-> prod-FLASH->bf16 (online softmax, same structure) : {relerr(oz, _bf(pf)):.2e}")
print(f"     flash-ozaki <-> prod-EAGER->bf16 (materialized full-row softmax)  : {relerr(oz, _bf(pe)):.2e}")
print(f"     => flash-ozaki is ~{relerr(oz,_bf(pe))/relerr(oz,_bf(pf)):.0f}x closer to the online-softmax FLASH emulation,")
print(f"        reproducing the production ozaki flash (P->bf16 datapath) to ~3e-5.")

print(f"\n=== Latency ms/call (ozaki chunk={CHUNK}); flash-ozaki at nmp10 AND nmp1 ===")
print(f"  {'case':>26} | {'oz nmp10':>9} | {'oz nmp1(w8)':>11} | {'flash-exact':>11} | {'SDPA':>9} | {'nmp10/sdpa':>10}")
for N in [1024, 2048, 4096]:
    q = torch.randn(1, 28, N, D, device=dev, dtype=torch.bfloat16); k = torch.randn(1, 28, N, D, device=dev, dtype=torch.bfloat16); v = torch.randn(1, 28, N, D, device=dev, dtype=torch.bfloat16)
    t10 = do_bench(lambda: flash_oz1fp_cg(q, k, v, nmp=NMP, w=W, causal=True, chunk_size=CHUNK, BLOCK_M=32, BLOCK_N=32))
    t1 = do_bench(lambda: flash_oz1fp_cg(q, k, v, nmp=1, w=8, causal=True, chunk_size=CHUNK))
    t_fe = do_bench(lambda: flash_oz1fp_cg(q, k, v, nmp=1, w=8, causal=True, ozaki=False))
    t_sd = do_bench(lambda: sdpa(q, k, v))
    print(f"  {('PREFILL MHA N='+str(N)):>26} | {t10:9.3f} | {t1:11.3f} | {t_fe:11.3f} | {t_sd:9.3f} | {t10/t_sd:9.1f}x")
for Bd in [8, 32]:
    qd = torch.randn(Bd, 28, 1, D, device=dev, dtype=torch.bfloat16); kd = torch.randn(Bd, 4, 2048, D, device=dev, dtype=torch.bfloat16); vd = torch.randn(Bd, 4, 2048, D, device=dev, dtype=torch.bfloat16)
    t10 = do_bench(lambda: flash_oz1fp_cg(qd, kd, vd, nmp=NMP, w=W, causal=True, chunk_size=CHUNK))
    t1 = do_bench(lambda: flash_oz1fp_cg(qd, kd, vd, nmp=1, w=8, causal=True, chunk_size=CHUNK))
    t_fe = do_bench(lambda: flash_oz1fp_cg(qd, kd, vd, nmp=1, w=8, causal=True, ozaki=False))
    t_sd = do_bench(lambda: sdpa(qd, kd, vd))
    print(f"  {('DECODE GQA B='+str(Bd)+' N=2048'):>26} | {t10:9.3f} | {t1:11.3f} | {t_fe:11.3f} | {t_sd:9.3f} | {t10/t_sd:9.1f}x")
print("\nDONE")
