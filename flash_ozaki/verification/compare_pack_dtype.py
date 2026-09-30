"""Part H.5/H.6: bf16 packing (absolute place, g=8//w) vs fp16 packing (relative place + external
place_exp, g=11//w). fp16 has 11 significand bits so w=5 gets g=2 back -- 4x fewer dots at nmp 4/16.

Run at GEMM level on purpose: the flash path stores bf16, which hides differences below its ULP.
`exact` here is the same digit polynomial evaluated in int64/fp64 -- what BOTH kernels intend to
compute -- so "closer to exact" means the more faithful emulation.

  python flash_ozaki/verification/compare_pack_dtype.py
"""
import os, sys
import torch
sys.path.insert(0, os.path.abspath("."))
from flash_ozaki.verification.standalone_oz_gemm import oz1fp_gemm_cg
from flash_ozaki.flash_oz1fp_codegen import pack_plan, _bfp_scale_torch, _digit_planes
from flash_ozaki.oz1fp_triton import oz1fp_params
from emulation.llm.ozaki_matmul import ozaki1_batched_gemm_fp, CustomGemmConfig, Ozaki1Config

torch.cuda.set_device(0)
STYLE, CHUNK = "all_signed_no_clamp", 32
Z, M, K, N = 1, 128, 128, 128
g = torch.Generator(device="cuda").manual_seed(0)
A = torch.randn(Z, M, K, generator=g, device="cuda", dtype=torch.bfloat16)
B = torch.randn(Z, K, N, generator=g, device="cuda", dtype=torch.bfloat16)
gcfg = CustomGemmConfig(in_feature_ts=K, out_feature_ts=N, chunk_size=CHUNK, name="cmp",
                        track_mtx_acc=False, track_model_acc=False, get_statistics=False,
                        rslt_type="ozaki1_fp")
d = lambda x, y: ((x.double() - y.double()).norm() / (y.double().norm() + 1e-30)).item()


def exact_poly(w, nmp):
    """the digit polynomial evaluated EXACTLY (int64 per chunk, fp64 combine) -- ground truth for
    what BOTH kernels are supposed to compute"""
    nD, drop, ib = oz1fp_params(nmp, w)
    out = torch.zeros(Z, M, N, dtype=torch.float64, device="cuda")
    for c0 in range(0, K, CHUNK):
        a = A[:, :, c0:c0 + CHUNK].float(); b = B[:, c0:c0 + CHUNK, :].float()
        sa = _bfp_scale_torch(a.abs().amax(dim=2), ib)
        sb = _bfp_scale_torch(b.abs().amax(dim=1), ib)
        aI = (a / sa[..., None] + torch.where(a >= 0, 0.5, -0.5)).to(torch.int64).clamp(-(1 << ib), (1 << ib) - 1)
        bI = (b / sb[:, None, :] + torch.where(b >= 0, 0.5, -0.5)).to(torch.int64).clamp(-(1 << ib), (1 << ib) - 1)
        # raw signed digits (undo the place fold that _digit_planes applies)
        da = [(_digit_planes(aI, w, nD)[:, t].double() / 2.0 ** (w * t)) for t in range(nD)]
        db = [(_digit_planes(bI.transpose(1, 2).contiguous(), w, nD)[:, t].double()
               / 2.0 ** (w * t)).transpose(1, 2) for t in range(nD)]
        acc = torch.zeros(Z, M, N, dtype=torch.float64, device="cuda")
        for i in range(nD):
            for j in range(nD):
                if w * (i + j) >= drop:
                    acc += (da[i] @ db[j]) * float(1 << (w * (i + j)))
        out += acc * sa[..., None].double() * sb[:, None, :].double()
    return out


print("=" * 104)
print("bf16 (absolute place, g=8//w) vs fp16 (relative place + external place_exp, g=11//w)")
print("=" * 104)
print(f"{'w':>3}{'nmp':>5}{'int_bits':>9}{'dots bf/fp':>12}{'bf16 vs fp16':>15}"
      f"{'bf16 vs exact':>15}{'fp16 vs exact':>15}{'prod vs exact':>15}")
for w in (4, 5):
    for nmp in (1, 3, 4, 6, 9, 10, 15, 16):
        nD, _, ib = oz1fp_params(nmp, w)
        ex = exact_poly(w, nmp)
        a = oz1fp_gemm_cg(A, B, nmp=nmp, w=w, chunk=CHUNK, byte_split_style=STYLE, pack_dtype="bf16")
        b = oz1fp_gemm_cg(A, B, nmp=nmp, w=w, chunk=CHUNK, byte_split_style=STYLE, pack_dtype="fp16")
        oz1 = Ozaki1Config(rounding="round_half_away_from_0", nmp=nmp, byte_split_style=STYLE)
        pr = ozaki1_batched_gemm_fp(A, B, gcfg, oz1, out_dtype=torch.float32, gemm_bits=w)
        na, nb = len(pack_plan(nmp, w, "bf16")), len(pack_plan(nmp, w, "fp16"))
        eq = "BIT-EQ" if torch.equal(a, b) else f"{d(a, b):.2e}"
        print(f"{w:>3}{nmp:>5}{ib:>9}{f'{na}/{nb}':>12}{eq:>15}"
              f"{d(a, ex):>15.2e}{d(b, ex):>15.2e}{d(pr, ex):>15.2e}")
print("\n  'exact' = the same digit polynomial evaluated in int64/fp64 -- what both kernels intend to")
print("  compute. Lower vs exact = the more faithful emulation.")
