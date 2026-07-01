"""Faithfulness of the flash-ozaki codegen encode vs production ozaki1_batched_gemm_fp:
  (1) block-FP scale convention matches (frexp + int_bits=w*nD-1) -> dequantized operand Â bit-identical;
  (2) no_clamp top digit lands in [-2^(w-1), 2^(w-1)] (the int8 datapath range + 1-bit MSB flag);
  (3) signed digit-split is identical to production's _oz1_wbit_digit_split;
  (4) the kernel result == its own quantized product (fp64) to the fp32-accumulation floor.
Run from repo root:  CUDA_VISIBLE_DEVICES=2 python flash_ozaki/verification/verify_faithfulness.py"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/howonlee/Quantized-Reasoning-Models")
from standalone_oz_gemm import oz1fp_gemm_cg, _bfp_scale_torch
from flash_ozaki.oz1fp_triton import oz1fp_params
from emulation.llm.ozaki_matmul import _oz1_wbit_encode, _oz1_wbit_digit_split
dev = "cuda"; torch.manual_seed(0)
rnd = lambda x: torch.trunc(x + torch.where(x >= 0, 0.5, -0.5))
def rel(a, b): return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


def mine_hatA(A, w, nD, ch):                                # production-faithful encode of A (operand)
    ib = w * nD - 1; Z, M, K = A.shape; nc = K // ch
    Ar = A.float().reshape(Z, M, nc, ch)
    sA = _bfp_scale_torch(Ar.abs().amax(-1, keepdim=True), ib)
    aI = rnd(Ar / sA).clamp(-(1 << ib), (1 << ib) - 1)
    return (aI * sA).reshape(Z, M, K)


def prod_hatA(A, w, nD, ch):
    ib = w * nD - 1
    dv, Xi, _ = _oz1_wbit_encode(A, ch, -1, ib, w, 'round_half_away_from_0', 'A', no_clamp=True, predicate='pos')
    Z, M, K = A.shape
    return (Xi.float() * torch.exp2(-dv.float())).permute(1, 2, 0, 3).reshape(Z, M, K)


print("=== (1) dequantized operand Â: flash-codegen-encode vs production ===")
Z, M, K, ch = 4, 128, 128, 32
for (w, nmp) in [(4, 9), (4, 16), (8, 1), (2, 16)]:
    nD = oz1fp_params(nmp, w)[0]
    A = torch.randn(Z, M, K, device=dev, dtype=torch.bfloat16) * 3
    Am, Ap = mine_hatA(A, w, nD, ch), prod_hatA(A, w, nD, ch)
    print(f"  w{w} nmp{nmp:<2}: Â equal={torch.equal(Am, Ap)}  relerr={rel(Am, Ap):.2e}")

print("\n=== (2) no_clamp top-digit range (should be [-2^(w-1), 2^(w-1)] = int8 + 1-bit flag) ===")
def peel_top(Xi, w, nD):
    base, half = (1 << w), (1 << (w - 1)); cur = Xi.clone()
    for _ in range(nD - 1):
        low = cur & (base - 1); cur = (cur - torch.where(low >= half, low - base, low)) >> w
    return cur
for (w, nmp) in [(4, 9), (8, 1), (2, 16)]:
    nD = oz1fp_params(nmp, w)[0]; ib = w * nD - 1
    A = torch.randn(Z, M, K, device=dev, dtype=torch.bfloat16) * 3
    Ar = A.float().reshape(Z, M, K // ch, ch); sA = _bfp_scale_torch(Ar.abs().amax(-1, keepdim=True), ib)
    Xi = rnd(Ar / sA).clamp(-(1 << ib), (1 << ib) - 1).to(torch.int64).reshape(Z, M, K)
    top = peel_top(Xi, w, nD)
    print(f"  w{w} nmp{nmp:<2}: top digit in [{top.min().item()}, {top.max().item()}]  (expect [{-(1<<(w-1))}, {1<<(w-1)}])")

print("\n=== (3) signed digit-split: flash peel vs production _oz1_wbit_digit_split (same Xi) ===")
def mine_split(Xi, w, nD, nc_):
    base, half = (1 << w), (1 << (w - 1)); cur = Xi.clone(); ds = []
    for t in range(nD):
        if t == nD - 1 and nc_: d = cur
        else:
            low = cur & (base - 1); d = torch.where(low >= half, low - base, low); cur = (cur - d) >> w
        ds.append(d)
    return ds
for (w, nD) in [(4, 3), (4, 4), (8, 1), (2, 5)]:
    lim = 1 << (w * nD - 1); Xi = torch.randint(-lim, lim, (4096,), device=dev, dtype=torch.int64)
    for nc_ in [True, False]:
        eq = all(torch.equal(a, b) for a, b in zip(mine_split(Xi, w, nD, nc_), _oz1_wbit_digit_split(Xi, w, nD, no_clamp=nc_)))
        print(f"  w{w} nD{nD} no_clamp={int(nc_)}: digits identical = {eq}")

print("\n=== (4) flash GEMM result == its own quantized product (fp64) -> fp32-accumulation floor ===")
def mine_hatB(B, w, nD, ch):
    ib = w * nD - 1; Z, K, N = B.shape; nc = K // ch
    Br = B.float().reshape(Z, nc, ch, N); sB = _bfp_scale_torch(Br.abs().amax(2, keepdim=True), ib)
    return (rnd(Br / sB).clamp(-(1 << ib), (1 << ib) - 1) * sB).reshape(Z, K, N)
Z, M, K, N, ch = 4, 128, 128, 128, 32
for (w, nmp) in [(4, 9), (4, 16), (8, 1)]:
    nD = oz1fp_params(nmp, w)[0]
    A = torch.randn(Z, M, K, device=dev, dtype=torch.bfloat16); B = torch.randn(Z, K, N, device=dev, dtype=torch.bfloat16)
    out = oz1fp_gemm_cg(A, B, nmp, w, chunk=ch, byte_split_style="all_signed_no_clamp")
    C64 = mine_hatA(A, w, nD, ch).double() @ mine_hatB(B, w, nD, ch).double()
    print(f"  w{w} nmp{nmp:<2}: rel(kernel, Â@B̂ fp64) = {rel(out, C64):.2e}")
print("\nDONE")
