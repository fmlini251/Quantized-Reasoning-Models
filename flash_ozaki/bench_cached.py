"""Pre-encoded KV cache (attention analog of weight_cache), EXACT vs non-cached ozaki1_fp (codegen
path): cache format/memory + accuracy (vs fp32 exact and vs the non-cached kernel), prefill + decode
(MHA and GQA head-fold). K/V are block-FP-encoded ONCE into place-folded digit planes; the flash
kernel loads them and skips the per-tile K/V amax+round+split+cast. Run from repo root:
    CUDA_VISIBLE_DEVICES=0 python flash_ozaki/bench_cached.py"""
import sys
import torch
sys.path.insert(0, "/home/howonlee/Quantized-Reasoning-Models")
from flash_ozaki.flash_oz1fp_codegen import flash_oz1fp_cg, flash_oz1fp_cg_cached, encode_kv
from flash_ozaki.oz1fp_triton import oz1fp_params
from flash_ozaki._testutil import relerr, exact_attn, do_bench, cache_mb


dev = "cuda"; torch.manual_seed(0)
B, H, D, N = 1, 28, 128, 2048
CHUNK = 32                                                    # block-FP reduction chunk (== production)
q = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16)
ex = exact_attn(q, k, v, True)

print(f"=== EXACT digit-plane KV cache (N={N}, H={H}, D={D}, causal, chunk={CHUNK}) ===")
print(f"{'cfg':>9} | {'planes(K/V)':>11} | {'cache MB':>8} | {'cached vs EXACT':>15} | "
      f"{'cached vs non-cache':>19} | {'ms cached/non-cache':>19}")
for (w, nmp) in [(4, 9), (4, 16), (4, 10), (4, 15)]:
    nD, drop, _ = oz1fp_params(nmp, w)
    kv = encode_kv(k, v, nmp, w, chunk_size=CHUNK)            # (k_pl, k_scale, v_pl, v_scale), exact planes
    o_c = flash_oz1fp_cg_cached(q, kv, nmp, w, chunk_size=CHUNK)
    nc = flash_oz1fp_cg(q, k, v, nmp, w, chunk_size=CHUNK)    # non-cached exact ozaki1_fp
    t_c = do_bench(lambda: flash_oz1fp_cg_cached(q, kv, nmp, w, chunk_size=CHUNK))
    t_nc = do_bench(lambda: flash_oz1fp_cg(q, k, v, nmp, w, chunk_size=CHUNK))
    print(f"  w{w} nmp{nmp:<2} | {nD:5}/{nD:<5} | {cache_mb(kv):8.1f} | {relerr(o_c,ex):15.2e} | "
          f"{relerr(o_c,nc):19.2e} | {t_c:8.2f}/{t_nc:8.2f}")
print("\ncached == non-cached bit-exact at chunk=None/decode; ~1e-5 (fp accumulation order of the two "
      "distinct compiled kernels, NOT a value diff) for chunked prefill -- see verification/RESULTS.md.")

# --- decode (q_len=1) against the cache: MHA + GQA fold (the cache is the real decode win) ---
Hkv = 4; G = H // Hkv                                         # Qwen-7B GQA shape (28/4 -> G=7)
print(f"\n=== Decode q_len=1 against the cache (N={N}, w4 nmp9, chunk={CHUNK}) ===")
qd = torch.randn(B, H, 1, D, device=dev, dtype=torch.bfloat16)                # 28 query heads, 1 token
ex_d = exact_attn(qd, k, v, True)                                            # MHA exact decode
kg = k.reshape(B, Hkv, G, N, D)[:, :, 0]; vg = v.reshape(B, Hkv, G, N, D)[:, :, 0]   # 4 kv heads
ex_g = exact_attn(qd, kg, vg, True)                                          # GQA exact (fold-aware)
o_mha = flash_oz1fp_cg_cached(qd, encode_kv(k, v, 9, 4, chunk_size=CHUNK), 9, 4, chunk_size=CHUNK)
o_gqa = flash_oz1fp_cg_cached(qd, encode_kv(kg, vg, 9, 4, chunk_size=CHUNK), 9, 4, chunk_size=CHUNK)
print(f"  MHA decode (cache 28 kv heads): cached vs EXACT = {relerr(o_mha, ex_d):.2e}")
print(f"  GQA decode (cache  4 kv heads, G=7 fold): cached vs EXACT = {relerr(o_gqa, ex_g):.2e}")
