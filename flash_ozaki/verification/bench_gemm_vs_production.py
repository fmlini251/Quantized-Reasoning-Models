"""GEMM-part comparison: the standalone QK (Q@K^T) and PV (P@V) ozaki1_fp matmuls (extracted from the
flash-attention kernel, codegen plan+peel) vs production ozaki1_batched_gemm_fp. cached/non-cached x
nmp x w. K-chunk = 32 (block-FP chunk, matching real configs). Reports: standalone-vs-prod relerr,
both vs fp32-exact, bit-exact (torch.equal), latency.  Run from the repo root:
    CUDA_VISIBLE_DEVICES=2 python flash_ozaki/verification/bench_gemm_vs_production.py"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # for standalone_oz_gemm
sys.path.insert(0, "/home/howonlee/Quantized-Reasoning-Models")
from standalone_oz_gemm import oz1fp_gemm_cg, encode_B
from flash_ozaki._testutil import relerr, do_bench as bench
from emulation.llm.ozaki_matmul import ozaki1_batched_gemm_fp, CustomGemmConfig, Ozaki1Config
dev = "cuda"; torch.manual_seed(0)
STYLE = "all_signed_no_clamp"
CHUNK = 32


def cfgs_for(K, N):
    g = CustomGemmConfig(in_feature_ts=K, out_feature_ts=N, chunk_size=CHUNK, name="cmp",
                         track_mtx_acc=False, track_model_acc=False, get_statistics=False,
                         rslt_type="ozaki1_fp")
    return g


def compare(tag, A, B, cfg_list):
    exact = A.float() @ B.float()
    Z, M, K = A.shape; N = B.shape[2]
    gcfg = cfgs_for(K, N)
    print(f"\n=== {tag}  A{tuple(A.shape)} @ B{tuple(B.shape)}  chunk={CHUNK} ===")
    print(f"  {'w/nmp':>9} {'cache':>5} | {'mine/prod':>10} | {'biteq':>5} | "
          f"{'mine/exact':>11} | {'prod/exact':>11} | {'mine ms':>8} | {'prod ms':>8} | {'x':>5}")
    for (w, nmp) in cfg_list:
        oz1 = Ozaki1Config(rounding="round_half_away_from_0", nmp=nmp, byte_split_style=STYLE)
        for cached in (False, True):
            try:
                if cached:
                    bc = encode_B(B, nmp, w, CHUNK, 1)
                    mine = oz1fp_gemm_cg(A, B, nmp, w, chunk=CHUNK, byte_split_style=STYLE, b_cache=bc, num_stages=2)
                    _, wc = ozaki1_batched_gemm_fp(A, B, gcfg, oz1, weight_cache=None,
                                                   return_weight_cache=True, out_dtype=torch.float32, gemm_bits=w)
                    prod = ozaki1_batched_gemm_fp(A, B, gcfg, oz1, weight_cache=wc, out_dtype=torch.float32, gemm_bits=w)
                    t_m = bench(lambda: oz1fp_gemm_cg(A, B, nmp, w, chunk=CHUNK, byte_split_style=STYLE, b_cache=bc, num_stages=2))
                    t_p = bench(lambda: ozaki1_batched_gemm_fp(A, B, gcfg, oz1, weight_cache=wc, out_dtype=torch.float32, gemm_bits=w))
                else:
                    mine = oz1fp_gemm_cg(A, B, nmp, w, chunk=CHUNK, byte_split_style=STYLE, num_stages=2)
                    prod = ozaki1_batched_gemm_fp(A, B, gcfg, oz1, out_dtype=torch.float32, gemm_bits=w)
                    t_m = bench(lambda: oz1fp_gemm_cg(A, B, nmp, w, chunk=CHUNK, byte_split_style=STYLE, num_stages=2))
                    t_p = bench(lambda: ozaki1_batched_gemm_fp(A, B, gcfg, oz1, out_dtype=torch.float32, gemm_bits=w))
                biteq = torch.equal(mine.float(), prod.float())
                print(f"  w{w} nmp{nmp:<2} {'cache' if cached else 'fresh':>5} | {relerr(mine,prod):10.2e} | "
                      f"{str(biteq):>5} | {relerr(mine,exact):11.2e} | {relerr(prod,exact):11.2e} | "
                      f"{t_m:8.3f} | {t_p:8.3f} | {t_p/t_m:4.2f}x")
            except Exception as e:
                print(f"  w{w} nmp{nmp:<2} {'cache' if cached else 'fresh':>5} | ERROR: {str(e)[:55]}")


CFGS = [(8, 1), (4, 9), (4, 10), (4, 15), (4, 16)]
Z = 28; D = 128
# prefill-ish (square)
Mq = Mk = 1024
Q = torch.randn(Z, Mq, D, device=dev, dtype=torch.bfloat16)
Kt = torch.randn(Z, D, Mk, device=dev, dtype=torch.bfloat16)
compare("PREFILL QK = Q @ K^T (reduction D=128)", Q, Kt, CFGS)
P = torch.rand(Z, Mq, Mk, device=dev, dtype=torch.bfloat16)
V = torch.randn(Z, Mk, D, device=dev, dtype=torch.bfloat16)
compare("PREFILL PV = P @ V (reduction Mk=1024)", P, V, CFGS)
# decode-ish (small M: one folded query tile, long kv)
Mqd, Mkd = 16, 2048
Qd = torch.randn(Z, Mqd, D, device=dev, dtype=torch.bfloat16)
Ktd = torch.randn(Z, D, Mkd, device=dev, dtype=torch.bfloat16)
compare("DECODE  QK = Q @ K^T (Mq=16, reduction D=128)", Qd, Ktd, CFGS)
Pd = torch.rand(Z, Mqd, Mkd, device=dev, dtype=torch.bfloat16)
Vd = torch.randn(Z, Mkd, D, device=dev, dtype=torch.bfloat16)
compare("DECODE  PV = P @ V (Mq=16, reduction Mk=2048)", Pd, Vd, CFGS)
print("\nDONE")
