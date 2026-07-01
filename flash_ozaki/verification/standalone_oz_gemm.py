"""Standalone ozaki1_fp GEMM (A@B) extracted from the flash-attention QK/PV machinery (codegen:
optimal pack plan + single-peel place-folded planes + production-faithful block-FP scale). Used by
bench_gemm_vs_production.py to compare the attention kernel's isolated QK (Q@K^T) and PV (P@V) matmuls
against production ozaki1_batched_gemm_fp. The block-FP scale matches production exactly (frexp +
int_bits=w*nD-1, clamp to [-2^int_bits, 2^int_bits-1]) so the dequantized operands are bit-identical
to production and the residual is purely fp32 accumulation order."""
import sys, linecache
import torch, triton, triton.language as tl
sys.path.insert(0, "/home/howonlee/Quantized-Reasoning-Models")
from flash_ozaki.flash_oz1fp_codegen import (pack_plan, _emit_peel, _super, _emit_dots, _bfp_scale,
                                             _bfp_scale_torch, _digit_planes)
from flash_ozaki.oz1fp_triton import oz1fp_params
# _bfp_scale_torch / _digit_planes live in flash_oz1fp_codegen (single source); re-exported here so
# verify_faithfulness.py's `from standalone_oz_gemm import _bfp_scale_torch` keeps working.


def _gen_gemm_src(nmp, w, no_clamp, cached):
    nD = oz1fp_params(nmp, w)[0]
    ib = w * nD - 1                                          # production int_bits = w*nD-1
    plan = pack_plan(nmp, w)
    L = "        "
    # B operand: cached -> load nD pre-encoded place-folded planes Bp[t]; else encode inline.
    if cached:
        bload = "\n".join(                                       # planes named bpp{t} to match _super("bp",..)
            f"{L}bpp{t} = tl.load(Bp + pid_z*sbpz + {t}*sbpt + offk[:,None]*sbpk + offn[None,:]*sbpn,"
            f" mask=km[:,None] & nm[None,:], other=0.0)" for t in range(nD))
        bargs = "Bp, Bs, "
        bstrides = "sbpz, sbpt, sbpk, sbpn, sbsz, sbsc, sbsn, "
        bscale = f"{L}sB = tl.load(Bs + pid_z*sbsz + (kc//CHUNK)*sbsc + offn*sbsn, mask=nm, other=0.0)"
    else:
        bload = (f"{L}b = tl.load(B + pid_z*sbz + offk[:,None]*sbk + offn[None,:]*sbn,"
                 f" mask=km[:,None] & nm[None,:], other=0.0)\n"
                 f"{L}sB = _bfp_scale(tl.max(tl.abs(b), axis=0).to(tl.float32), {ib})\n"
                 f"{L}bI = (b / sB[None,:] + tl.where(b>=0, 0.5, -0.5)).to(tl.int32)\n"
                 f"{L}bI = tl.minimum(tl.maximum(bI, {-(1<<ib)}), {(1<<ib)-1})\n"
                 + _emit_peel("bp", "bI", w, nD, no_clamp, L))
        bargs = "B, "
        bstrides = "sbz, sbk, sbn, "
        bscale = ""
    src = f'''
@triton.jit
def _gemm_cg(A, {bargs}C, M, N, K,
    saz, sam, sak, {bstrides}scz, scm, scn,
    CHUNK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1); pid_z = tl.program_id(2)
    offm = pid_m*BM + tl.arange(0, BM); offn = pid_n*BN + tl.arange(0, BN)
    mm_ = offm < M; nm = offn < N
    acc = tl.zeros([BM, BN], tl.float32)
    for kc in range(0, K, CHUNK):
        offk = kc + tl.arange(0, CHUNK); km = offk < K
        a = tl.load(A + pid_z*saz + offm[:,None]*sam + offk[None,:]*sak, mask=mm_[:,None] & km[None,:], other=0.0)
        sA = _bfp_scale(tl.max(tl.abs(a), axis=1).to(tl.float32), {ib})
        aI = (a / sA[:,None] + tl.where(a>=0, 0.5, -0.5)).to(tl.int32)
        aI = tl.minimum(tl.maximum(aI, {-(1<<ib)}), {(1<<ib)-1})
{_emit_peel("ap", "aI", w, nD, no_clamp, L)}
{bscale}
{bload}
        cacc = tl.zeros([BM, BN], tl.float32)
{_emit_dots("cacc", plan, "ap", "bp", False, L)}
        acc += cacc * sA[:,None] * sB[None,:]
    tl.store(C + pid_z*scz + offm[:,None]*scm + offn[None,:]*scn, acc, mask=mm_[:,None] & nm[None,:])
'''
    return src


_CACHE = {}


def _get(nmp, w, no_clamp, cached):
    key = (nmp, w, no_clamp, cached)
    if key not in _CACHE:
        src = _gen_gemm_src(*key); fn = f"<gemm_cg_{key}>"
        linecache.cache[fn] = (len(src), None, src.splitlines(keepends=True), fn)
        ns = {"triton": triton, "tl": tl, "_bfp_scale": _bfp_scale}
        exec(compile(src, fn, "exec"), ns)
        _CACHE[key] = ns["_gemm_cg"]
    return _CACHE[key]


def encode_B(B, nmp, w, chunk, no_clamp):
    """B:[Z,K,N] -> (Bp[Z,nD,K,N] place-folded bf16 planes, Bs[Z,nchunk,N] per-(chunk,col) scale)."""
    Z, K, N = B.shape
    nD = oz1fp_params(nmp, w)[0]; ib = w * nD - 1
    rnd = lambda x: torch.trunc(x + torch.where(x >= 0, 0.5, -0.5))
    nch = (K + chunk - 1) // chunk
    Bs = torch.empty(Z, nch, N, device=B.device, dtype=torch.float32)
    Xi = torch.empty(Z, K, N, device=B.device, dtype=torch.float32)
    bf = B.float()
    for c in range(nch):
        s, e = c*chunk, min((c+1)*chunk, K)
        sc = _bfp_scale_torch(bf[:, s:e, :].abs().amax(1, keepdim=True), ib)
        Bs[:, c, :] = sc[:, 0, :]
        Xi[:, s:e, :] = rnd(bf[:, s:e, :] / sc).clamp(-(1 << ib), (1 << ib) - 1)
    Bp = _digit_planes(Xi.to(torch.int64), w, nD, bool(no_clamp))   # [Z,nD,K,N] place-folded bf16
    return Bp.contiguous(), Bs.contiguous()


def oz1fp_gemm_cg(A, B, nmp, w, chunk=None, byte_split_style="all_signed_no_clamp",
                  b_cache=None, BM=64, BN=64, num_warps=4, num_stages=2):
    """A:[Z,M,K] @ B:[Z,K,N] -> [Z,M,N] fp32, ozaki1_fp via codegen plan+peel. b_cache=encode_B(...)
    for the cached path (B pre-encoded, like weight_cache)."""
    Z, M, K = A.shape
    N = B.shape[2] if B is not None else b_cache[0].shape[3]
    no_clamp = 1 if byte_split_style == "all_signed_no_clamp" else 0
    CHUNK = K if chunk is None else min(chunk, K)
    cached = b_cache is not None
    C = torch.empty(Z, M, N, device=A.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), Z)
    kern = _get(nmp, w, no_clamp, cached)
    if cached:
        Bp, Bs = b_cache
        kern[grid](A, Bp, Bs, C, M, N, K,
                   A.stride(0), A.stride(1), A.stride(2),
                   Bp.stride(0), Bp.stride(1), Bp.stride(2), Bp.stride(3),
                   Bs.stride(0), Bs.stride(1), Bs.stride(2),
                   C.stride(0), C.stride(1), C.stride(2),
                   CHUNK=CHUNK, BM=BM, BN=BN, num_warps=num_warps, num_stages=num_stages)
    else:
        kern[grid](A, B, C, M, N, K,
                   A.stride(0), A.stride(1), A.stride(2),
                   B.stride(0), B.stride(1), B.stride(2),
                   C.stride(0), C.stride(1), C.stride(2),
                   CHUNK=CHUNK, BM=BM, BN=BN, num_warps=num_warps, num_stages=num_stages)
    return C
