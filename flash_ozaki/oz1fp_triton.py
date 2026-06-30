"""bf16 byte-plane Ozaki-1 fp GEMM in Triton (the `ozaki1_fp` mechanism, NOT int8).

Each operand is block-FP scaled, rounded to an integer, and split into `nD` signed w-bit digits
held as **bf16** values (small ints, exact in bf16). The product is the sum over kept digit-pairs
(la,lb) of bf16 matmuls weighted by the place value 2^(w*(la+lb)), accumulated in fp32, then
rescaled by the block-FP scales. This is exactly what the emulation's ozaki1_fp does, just fused
in Triton (bf16 tensor cores, fp32 accumulate -> the digit-pair products are exact).

nmp -> (nD, full|triangular):  full nmp=nD^2 (keep all pairs); triangular nmp=nD(nD+1)/2 (keep
high-significance pairs w*(la+lb) >= w*(nD-1)).  int_bits = w*nD-1.  (w=4 nmp=4 == w=8 nmp=1.)
Block-FP granularity here = per-row over the full reduction (the production emulation chunks by
k=32; a fidelity refinement, same mechanism). Digit split = signed carry split (clamp family);
the sweep uses no_clamp, a ~boundary-digit difference, validated small below.
"""
import math
import torch
import triton
import triton.language as tl


def oz1fp_params(nmp, w):
    r = math.isqrt(nmp)
    if r * r == nmp:                       # full polynomial
        nD, drop = r, 0
    else:                                  # triangular truncation
        nD = (math.isqrt(8 * nmp + 1) - 1) // 2
        assert nD * (nD + 1) // 2 == nmp, f"nmp={nmp} is neither full (nD^2) nor triangular (T_nD)"
        drop = w * (nD - 1)
    return nD, drop, w * nD - 1            # nD, drop_exp, int_bits


@triton.jit
def _signed_digit(Xi, t: tl.constexpr, W: tl.constexpr, ND: tl.constexpr):
    """t-th signed base-2^W digit of int tile Xi via the no_clamp carry split (low->high), matching
    byte_split_style='all_signed_no_clamp': digits 0..ND-2 land in [-2^(W-1), 2^(W-1)-1]; the TOP
    digit (t=ND-1) is the remaining value unclamped so it absorbs the carry (exact in bf16).
    Single return (constexpr ternary) -> avoids the divergent-return IR that failed to compile."""
    base = (1 << W); half = (1 << (W - 1))
    x = Xi
    for _ in range(t):
        low = x & (base - 1)
        dd = tl.where(low >= half, low - base, low)
        x = (x - dd) >> W
    low = x & (base - 1)
    clamped = tl.where(low >= half, low - base, low)
    return x if (t == ND - 1) else clamped         # constexpr condition picks one at compile time


@triton.jit
def _oz1fp_gemm(A, B, C, M, N, K,
                sa_z, sa_m, sa_k, sb_z, sb_k, sb_n, sc_z, sc_m, sc_n,
                W: tl.constexpr, ND: tl.constexpr, DROP: tl.constexpr, MAXMAG: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1); pid_z = tl.program_id(2)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    offk = tl.arange(0, BK)
    base = (1 << W); half = (1 << (W - 1))

    a = tl.load(A + pid_z * sa_z + offm[:, None] * sa_m + offk[None, :] * sa_k,
                mask=(offm[:, None] < M) & (offk[None, :] < K), other=0.0)   # [BM,BK] bf16
    b = tl.load(B + pid_z * sb_z + offk[:, None] * sb_k + offn[None, :] * sb_n,
                mask=(offk[:, None] < K) & (offn[None, :] < N), other=0.0)   # [BK,BN] bf16

    # block-FP scale (per A-row / per B-col over the reduction) and integer encode
    sA = tl.max(tl.abs(a), axis=1).to(tl.float32) / MAXMAG + 1e-20
    sB = tl.max(tl.abs(b), axis=0).to(tl.float32) / MAXMAG + 1e-20
    aI = (a / sA[:, None] + tl.where(a >= 0, 0.5, -0.5)).to(tl.int32)
    bI = (b / sB[None, :] + tl.where(b >= 0, 0.5, -0.5)).to(tl.int32)

    # ozaki1_fp: sum over kept digit-pairs (la,lb) of bf16 digit-dots, weighted by place value
    # 2^(W*(la+lb)), fp32-accumulated; ND/W/DROP constexpr so the loops unroll.
    # Fold the place value 2^(W*(la+lb)) into the digits as a power-of-2 scale BEFORE the dot:
    # digit*2^(W*la) is exact in bf16 (small int, exponent shift), and the per-operand shift
    # 1<<(W*la) <= 2^16 for our configs (w=4 nD<=5, w=8 nD=1) so no int32 overflow. The dot then
    # accumulates sum d_a*d_b*2^(W*(la+lb)) exactly in fp32 -- no post-dot float place needed.
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for la in range(ND):
        ad = (_signed_digit(aI, la, W, ND) * (1 << (W * la))).to(tl.bfloat16)
        for lb in range(ND):
            if W * (la + lb) >= DROP:
                bd = (_signed_digit(bI, lb, W, ND) * (1 << (W * lb))).to(tl.bfloat16)
                acc += tl.dot(ad, bd, out_dtype=tl.float32)
    acc = acc * sA[:, None] * sB[None, :]
    tl.store(C + pid_z * sc_z + offm[:, None] * sc_m + offn[None, :] * sc_n, acc,
             mask=(offm[:, None] < M) & (offn[None, :] < N))


def oz1fp_gemm(A, B, nmp, w, BM=64, BN=64):
    """A:[Z,M,K] @ B:[Z,K,N] (bf16) -> [Z,M,N] fp32 via bf16 byte-plane ozaki1_fp."""
    Z, M, K = A.shape
    _, _, N = B.shape
    nD, drop, int_bits = oz1fp_params(nmp, w)
    maxmag = float((1 << int_bits) - 1)
    C = torch.empty(Z, M, N, device=A.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), Z)
    _oz1fp_gemm[grid](A, B, C, M, N, K,
                      A.stride(0), A.stride(1), A.stride(2),
                      B.stride(0), B.stride(1), B.stride(2),
                      C.stride(0), C.stride(1), C.stride(2),
                      W=w, ND=nD, DROP=drop, MAXMAG=maxmag, BM=BM, BN=BN, BK=K,
                      num_warps=4, num_stages=2)
    return C


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/howonlee/Quantized-Reasoning-Models")
    from vllm_custom.model_executor.layers.ozaki_linear import build_ozaki_configs
    from emulation.llm.ozaki_matmul import batched_gemm
    import copy
    torch.manual_seed(0)
    dev = "cuda"

    def relerr(a, b):
        return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)

    Z, M, K, N = 2, 256, 128, 256
    A = torch.randn(Z, M, K, device=dev, dtype=torch.bfloat16)
    B = torch.randn(Z, K, N, device=dev, dtype=torch.bfloat16)
    exact = A.float() @ B.float()
    print("=== Triton bf16 ozaki1_fp GEMM vs production batched_gemm (and exact) ===")
    print(f"{'cfg':>14} | {'mine vs prod':>13} | {'mine vs exact':>13} | {'prod vs exact':>13}")
    for (w, nmp) in [(8, 1), (4, 4), (4, 9), (4, 10), (4, 15)]:
        gcfg, oz = build_ozaki_configs(nmp=nmp, rslt_type="ozaki1_fp", chunk_size=32,
                                       weight_cache=False, gemm_bits=w,
                                       byte_split_style="all_signed_no_clamp")
        cfg = copy.copy(gcfg); cfg.name = "probe"
        prod = batched_gemm(A, B, custom_gemm_config=cfg, ozaki_config=oz, out_dtype=torch.float32)
        mine = oz1fp_gemm(A, B, nmp=nmp, w=w)
        print(f"w={w} nmp={nmp:<2} (nD={oz1fp_params(nmp,w)[0]}) | {relerr(mine,prod):13.2e} | "
              f"{relerr(mine,exact):13.2e} | {relerr(prod,exact):13.2e}")
