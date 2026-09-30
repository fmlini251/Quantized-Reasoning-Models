"""Code-generated Flash-Ozaki1_fp: per-(nmp,w) the kernel source is emitted with (a) the OPTIMAL
signed-rectangle pack plan unrolled into explicit `tl.dot`s (min bf16-GEMM count, from the production
`_oz1_wbit_pack_plan`) and (b) a single low->high digit peel that emits NAMED place-folded planes
(`qp0..`, `kp0..`, ...) reused across the plan -- replacing the O(nD^2) repeated `_signed_digit`
re-peel + redundant place-mul/cast of the hand-rolled kernel.

Why codegen: the production GEMM avoids Triton's "can't index a constexpr tuple by a loop variable"
limitation by keeping the plan loop in host Python (one cuBLAS call per rectangle) and, for the only
fan-in kernel (`combine_cast`), flattening to fixed named slots G0..G5 + `if NMP>i` guards. A FUSED
flash kernel can do neither (online softmax forces QK->softmax->PV in one kernel). So we adopt the
same idea via codegen: emit each plan rectangle as its own explicit `tl.dot` statement and each digit
plane as its own named variable -- the fused-kernel analog of fixed named slots.

QK and PV are INDEPENDENT: each gets its own plan + planes + nmp/w. So QK-nmp != PV-nmp is free
(QK feeds softmax/exp -> precision-sensitive; PV is a weighted average -> often tolerates lower nmp).

This is the single kernel source for the flash_ozaki attention path:
  * flash_oz1fp_cg          -- fused flash attention (ozaki=True) or plain bf16 (ozaki=False baseline);
  * flash_oz1fp_cg_cached   -- the same, over a pre-encoded KV cache (encode_kv), the attention analog
                               of weight_cache: K/V are block-FP-encoded once, the kernel loads the
                               place-folded planes and skips the per-tile K/V encode;
  * _digit_planes / _bfp_scale_torch -- torch twins of the kernel's peel/scale, shared with the
                               standalone GEMM's weight-cache encoder (verification/standalone_oz_gemm).
On the production-faithful block-FP scale (frexp int_bits, clamp). Decode + GQA head-folding folds the
G group-heads into the query-row dim so one K/V tile feeds all G heads.
"""
import linecache
import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

from flash_ozaki.oz1fp_triton import oz1fp_params


def _run_with_oom_retry(launch, block_m):
    """Run launch(BLOCK_M); on shared-memory OutOfResources, halve BLOCK_M (down to 16) and retry.
    The cached kernel holds nD live K/V plane tiles, so a large-nD / triangular plan can exceed a
    GPU's shared-memory limit at BLOCK_M=64. Attention rows are independent (softmax is per row over
    kv), so BLOCK_M changes only occupancy/shared-mem -- never the result or the V block-FP chunking
    -- so shrinking it stays bit-exact vs the non-cached kernel."""
    while True:
        try:
            launch(block_m)
            return
        except OutOfResources:
            if block_m <= 16:
                raise
            block_m //= 2


@triton.jit
def _bfp_scale(amax, INT_BITS: tl.constexpr):
    """Production-faithful block-FP power-of-2 scale (matches calculate_scale_block_fp_torch):
    dv = INT_BITS - frexp_exp(amax); scale = 2^(-dv) = 2^(frexp_exp(amax) - INT_BITS), with INT_BITS =
    w*nD-1 (the canonical signed nD-digit w-bit range). frexp_exp(amax) = IEEE-754 fp32 exponent field
    - 126 (mantissa in [0.5,1)), extracted by bitcast -> EXACT (no float log2 rounding). This bounds
    |Xi| < 2^INT_BITS so the no_clamp top digit lands in [-2^(w-1), 2^(w-1)] (the int8 datapath range
    + 1-bit MSB flag), unlike the old _po2(ceil-log2, maxmag=2^(w*nD)-1) which over-shrank by ~1 bit."""
    a = tl.maximum(amax, 1e-30)
    e = ((a.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 126
    return tl.exp2((e - INT_BITS).to(tl.float32))


def _bfp_scale_torch(amax, int_bits):
    """Torch twin of _bfp_scale (production calculate_scale_block_fp_torch): 2^(frexp_exp - int_bits).
    Used to pre-encode a KV/weight cache off the kernel; the kernel then reloads the same scale."""
    _, e = torch.frexp(amax)
    return torch.exp2((e - int_bits).to(torch.float32))


def _digit_planes(xI, w, nD, no_clamp=True):
    """int64 xI[Z,...,K,N or N,D] -> [Z, nD, ...] bf16 planes, place-folded ({prefix}p{t} =
    signed_digit_t * 2^(w*t)). The torch twin of _emit_peel: SAME low->high signed peel and top-digit
    rule, so a cache encoded here is bit-consistent with the kernel's inline peel. Used by encode_kv
    (this module) and encode_B (standalone_oz_gemm)."""
    base, half = (1 << w), (1 << (w - 1))
    planes, cur = [], xI.clone()
    for t in range(nD):
        if t == nD - 1 and no_clamp:
            d = cur                                               # top digit = unclamped remainder
        else:
            lo = cur & (base - 1)
            d = torch.where(lo >= half, lo - base, lo)
            if t != nD - 1:
                cur = (cur - d) >> w
        planes.append((d.to(torch.float32) * (2.0 ** (w * t))).to(torch.bfloat16))
    return torch.stack(planes, dim=1)


# --- optimal pack plan (ported from emulation ozaki_matmul._oz1_wbit_pack_plan) -----------------
_PLAN_CACHE = {}


PACK_SIG = {"bf16": 8, "fp16": 11}          # significand bits (incl. the implicit one)
PACK_TL = {"bf16": "tl.bfloat16", "fp16": "tl.float16"}
PACK_TORCH = {"bf16": torch.bfloat16, "fp16": torch.float16}


def pack_plan(nmp, w, pack_dtype="bf16"):
    """Minimum signed g-bounded rectangle cover of the kept (la,lb) digit-pair region, with
    g = PACK_SIG[pack_dtype] // w. Returns [(i0,i1,j0,j1,sign)] -- each is one packed GEMM of
    A-digit-range [i0,i1] x B-digit-range [j0,j1] (span <= g -> the packed super-digit is exactly
    representable in the pack dtype), signed-combined.

    pack_dtype picks the packing width: bf16 has 8 significand bits (g = 8//w), fp16 has 11
    (g = 11//w). fp16 therefore recovers packing for the widths that do not divide 8 -- notably
    **w=5 goes g=1 -> g=2**, i.e. one dot per 2x2 digit-pair block instead of per cell. The exact
    bound is maxS(w,g) = 2^(w-1)*(2^(wg)-1)/(2^w-1) <= 2^SIG, and g = SIG//w happens to hit it
    exactly for every w in 2..8 (checked).

    Place convention depends on the dtype, because it decides the EXPONENT range needed:
      * bf16: absolute place 2^(w*t) folded into the planes -> no external weight (bf16 has fp32's
        exponent range, so a plane of 2^int_bits is free).
      * fp16: relative place inside the rectangle + external 2^(w*(i0+j0)) on the fp32 accumulator
        (production's convention). fp16 caps at 65504, so an absolute-place plane would go inf for
        int_bits >= 16; a relative super-digit only ever reaches maxS (<= 682).
    Found by IDA* (branch on rectangles covering the first uncovered cell, both signs)."""
    key = (nmp, w, pack_dtype)
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key]
    nD, drop, _ = oz1fp_params(nmp, w)
    g = max(1, PACK_SIG[pack_dtype] // w)
    kept = {(la, lb) for la in range(nD) for lb in range(nD) if w * (la + lb) >= drop}
    cells = [(la, lb) for la in range(nD) for lb in range(nD)]
    rects = []
    for i0 in range(nD):
        for i1 in range(i0, min(i0 + g, nD)):
            for j0 in range(nD):
                for j1 in range(j0, min(j0 + g, nD)):
                    cset = frozenset((i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1))
                    rects.append((i0, i1, j0, j1, cset))
    cover_by = {c: [r for r in rects if c in r[4]] for c in cells}
    ga = g * g
    sol = []

    def _lb(rem):
        nz = sum(1 for v in rem.values() if v)
        return (nz + ga - 1) // ga

    def _dfs(rem, depth, limit, seen):
        fc = next((c for c in cells if rem.get(c, 0) != 0), None)
        if fc is None:
            return True
        if depth + _lb(rem) > limit:
            return False
        state = (frozenset((c, v) for c, v in rem.items() if v), limit - depth)
        if state in seen:
            return False
        for r in cover_by[fc]:
            for sgn in (1, -1):
                nr = dict(rem)
                for c in r[4]:
                    nr[c] = nr.get(c, 0) - sgn
                sol.append((r, sgn))
                if _dfs(nr, depth + 1, limit, seen):
                    return True
                sol.pop()
        seen.add(state)
        return False

    target = {c: (1 if c in kept else 0) for c in cells}
    plan = None
    for limit in range(0, nD * nD + 1):
        sol.clear()
        if _dfs(dict(target), 0, limit, set()):
            plan = [(r[0], r[1], r[2], r[3], sgn) for (r, sgn) in sol]
            break
    chk = {c: 0 for c in cells}                                  # validate: signed cover == kept
    for (i0, i1, j0, j1, sgn) in plan:
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                chk[(i, j)] += sgn
    assert all(chk[c] == (1 if c in kept else 0) for c in cells), \
        f"bad plan nmp={nmp} w={w} pack={pack_dtype}"
    _PLAN_CACHE[key] = plan
    return plan


def peel_bias(w, nD):
    """The parallel balanced-digit peel's bias B = 2^(w-1) * (2^(w(nD-1))-1)/(2^w-1) (always an
    integer -- geometric series)."""
    return (1 << (w - 1)) * ((1 << (w * (nD - 1))) - 1) // ((1 << w) - 1) if nD > 1 else 0


def assert_int32_peel_fits(nmp, w):
    """The generated kernels clamp the block-FP integer to +-2^int_bits and peel it in **int32**, so
    2^int_bits + peel_bias must fit. int_bits = w*nD-1 grows fast: w=8 nD=4 gives 31 and w=8 nD=5
    gives 39, both of which overflow and produce SILENT garbage (measured vs production: w8 nmp10/16
    -> relerr 5.9e-2, w8 nmp15 -> 1.0, while production itself stays at 5e-8). Everything actually
    used here is far below the wall -- w=4 tops out at int_bits 19 (nmp15) and w=5 at 24 (nmp15) --
    so this only fences off the w=8 high-nD corner that was never exercised."""
    nD, _, int_bits = oz1fp_params(nmp, w)
    need = (1 << int_bits) + peel_bias(w, nD)
    if need >= 2 ** 31:
        raise ValueError(
            f"ozaki1_fp w={w} nmp={nmp} needs int_bits={int_bits} (nD={nD}); the int32 peel would "
            f"overflow (|z|max={need} >= 2^31) and return silently wrong results. Use a smaller w or "
            f"nmp: at w=8 keep nmp<=9 (int_bits<=23); w=4 (<=19) and w=5 (<=24) are always safe.")


# --- kernel source generation -----------------------------------------------------------------
def _emit_peel(prefix, src, w, nD, no_clamp, ind, pack_dtype="bf16"):
    """Emit a single low->high digit peel producing NAMED place-folded bf16 planes
    {prefix}p0..{prefix}p{nD-1}, where {prefix}p{t} = signed_digit_t * 2^(w*t) (absolute place folded
    in -> bf16-exact). One peel, O(nD) -- vs the hand-rolled kernel's O(nD^2) repeated _signed_digit.

    For the no_clamp datapath (the flash default, all_signed_no_clamp), the digits are emitted by the
    PARALLEL balanced-digit "bias trick" instead of the sequential borrow peel: with the constant
    B = 2^(w-1) * (2^(w(nD-1))-1)/(2^w-1) and z = src + B,
        d_t = ((z >> (w*t)) & (2^w-1)) - 2^(w-1)   (t < nD-1),   d_{nD-1} = z >> (w*(nD-1)).
    Each digit is an INDEPENDENT shift+and+sub (no `cur = (cur-d)>>w` borrow chain, no where-select),
    so the peel has fewer integer-ALU ops and full ILP -- ncu shows flash_ozaki is ALU-bound on exactly
    this integer work at low occupancy, and the parallel form is ~1.2-1.3x faster, BIT-IDENTICAL to the
    sequential peel (verified vs _digit_planes across w/nD incl. boundary values). The sequential peel is
    kept for the (non-flash) clamp styles where the top digit is clamped, not an unclamped remainder."""
    base, half = (1 << w), (1 << (w - 1))
    rel = pack_dtype != "bf16"          # relative place -> keep the digits as int32, _super folds+casts
    dt = PACK_TL[pack_dtype]
    if no_clamp:
        B = half * ((1 << (w * (nD - 1))) - 1) // (base - 1) if nD > 1 else 0
        L = [f"{ind}{prefix}_z = {src} + {B}"]
        for t in range(nD):
            if t == nD - 1:
                d = f"({prefix}_z >> {w * (nD - 1)})"             # top digit = unclamped remainder
            else:
                d = f"((({prefix}_z >> {w * t}) & {base - 1}) - {half})"
            L.append(f"{ind}{prefix}d{t} = {d}" if rel
                     else f"{ind}{prefix}p{t} = ({d} * {1 << (w * t)}).to({dt})")
        return "\n".join(L)
    L, cur = [], src
    for t in range(nD):
        L.append(f"{ind}{prefix}_lo = {cur} & {base - 1}")
        L.append(f"{ind}{prefix}d{t} = tl.where({prefix}_lo >= {half}, {prefix}_lo - {base}, {prefix}_lo)")
        d = f"{prefix}d{t}"
        if t != nD - 1:
            L.append(f"{ind}{prefix}_c{t} = ({cur} - {prefix}d{t}) >> {w}")
            cur = f"{prefix}_c{t}"
        if not rel:
            L.append(f"{ind}{prefix}p{t} = ({d} * {1 << (w * t)}).to({dt})")
    return "\n".join(L)


def _super(prefix, i0, i1, w=None, pack_dtype="bf16"):
    """Absolute convention (bf16): the place is already in the planes, so just add them.
    Relative convention (fp16): fold 2^(w*(t-i0)) in INT32 and cast ONCE -- multiplying an fp16
    tensor by a python float would promote the operand to fp32 and lose the fp16 tensor cores."""
    if pack_dtype == "bf16":
        return "(" + " + ".join(f"{prefix}p{t}" for t in range(i0, i1 + 1)) + ")"
    terms = [f"{prefix}d{t}" if t == i0 else f"{prefix}d{t} * {1 << (w * (t - i0))}"
             for t in range(i0, i1 + 1)]
    return "((" + " + ".join(terms) + f").to({PACK_TL[pack_dtype]}))"


def _emit_dots(acc, plan, a, b, trans_b, ind, w=None, pack_dtype="bf16"):
    """Emit one explicit signed tl.dot per plan rectangle (the fused-kernel analog of the GEMM's
    fixed named slots). bf16: place 2^(w*(i0+j0)) is already folded into the planes -> no external
    weight. fp16: apply it to the fp32 dot result (exact -- it is a power of two)."""
    L = []
    for (i0, i1, j0, j1, sgn) in plan:
        sa = _super(a, i0, i1, w, pack_dtype)
        sb = _super(b, j0, j1, w, pack_dtype)
        bexpr = f"tl.trans({sb})" if trans_b else sb
        op = "+=" if sgn > 0 else "-="
        # bf16: place already folded into the planes. otherwise weight the fp32 dot result by the
        # rectangle's place (exact -- a power of two); elide it when it is 1.
        pe = 0 if pack_dtype == "bf16" else w * (i0 + j0)
        scale = "" if pe == 0 else f" * {float(1 << pe)}"
        L.append(f"{ind}{acc} {op} tl.dot({sa}, {bexpr}, out_dtype=tl.float32){scale}")
    return "\n".join(L)


def _gen_src(nmp_qk, w_qk, nmp_pv, w_pv, no_clamp, chunked, pack_dtype="bf16"):
    assert_int32_peel_fits(nmp_qk, w_qk); assert_int32_peel_fits(nmp_pv, w_pv)
    nD_qk = oz1fp_params(nmp_qk, w_qk)[0]
    nD_pv = oz1fp_params(nmp_pv, w_pv)[0]
    ib_qk = w_qk * nD_qk - 1                                  # production int_bits = w*nD-1
    ib_pv = w_pv * nD_pv - 1
    plan_qk = pack_plan(nmp_qk, w_qk, pack_dtype)
    plan_pv = pack_plan(nmp_pv, w_pv, pack_dtype)
    I, L, L2 = "    ", "        ", "            "               # body / kv-loop / head-dim-chunk indents
    lo_qk, hi_qk = -(1 << ib_qk), (1 << ib_qk) - 1
    lo_pv, hi_pv = -(1 << ib_pv), (1 << ib_pv) - 1

    # QK block-FP scale granularity:
    #  * hoisted (chunk_size None / >= D): one scale over the full head_dim, Q peeled ONCE outside the
    #    kv loop and reused across tiles -- fastest, but coarser than production's chunk_size.
    #  * chunked (chunk_size < D): head_dim reduction split into KQ-wide block-FP chunks, per-chunk
    #    scale accumulated -- matches production's chunk_size exactly. Q is re-encoded per kv tile
    #    (per-chunk planes can't be hoisted across a runtime chunk loop), so it is slower.
    if not chunked:
        qk_pre = f'''    q = tl.load(Q + pid_z * sqz + offm[:, None] * sqn + offd[None, :] * sqd, mask=qmask, other=0.0)
    qsc = _bfp_scale(tl.max(tl.abs(q), axis=1).to(tl.float32), {ib_qk})
    qI = (q / qsc[:, None] + tl.where(q >= 0, 0.5, -0.5)).to(tl.int32)
    qI = tl.minimum(tl.maximum(qI, {lo_qk}), {hi_qk})   # block-FP clamp [-2^ib,2^ib-1]
{_emit_peel("q", "qI", w_qk, nD_qk, no_clamp, I, pack_dtype)}'''
        qk_body = f'''        k = tl.load(K + pid_z * skz + offn[:, None] * skn + offd[None, :] * skd,
                    mask=nmask[:, None] & dmask[None, :], other=0.0)
        ksc = _bfp_scale(tl.max(tl.abs(k), axis=1).to(tl.float32), {ib_qk})
        kI = (k / ksc[:, None] + tl.where(k >= 0, 0.5, -0.5)).to(tl.int32)
        kI = tl.minimum(tl.maximum(kI, {lo_qk}), {hi_qk})
{_emit_peel("k", "kI", w_qk, nD_qk, no_clamp, L, pack_dtype)}
        cacc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
{_emit_dots("cacc", plan_qk, "q", "k", True, L, w_qk, pack_dtype)}
        qk = cacc * qsc[:, None] * ksc[None, :] * sm_scale'''
    else:
        qk_pre = ""
        qk_body = f'''        qk = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for dc in range(0, HEAD_DIM, KQ):                         # chunk the head_dim reduction by KQ
            offc = dc + tl.arange(0, KQ)
            cmask = offc < HEAD_DIM
            qc = tl.load(Q + pid_z * sqz + offm[:, None] * sqn + offc[None, :] * sqd,
                         mask=mmask[:, None] & cmask[None, :], other=0.0)
            qsc = _bfp_scale(tl.max(tl.abs(qc), axis=1).to(tl.float32), {ib_qk})
            qI = (qc / qsc[:, None] + tl.where(qc >= 0, 0.5, -0.5)).to(tl.int32)
            qI = tl.minimum(tl.maximum(qI, {lo_qk}), {hi_qk})
{_emit_peel("q", "qI", w_qk, nD_qk, no_clamp, L2, pack_dtype)}
            kc = tl.load(K + pid_z * skz + offn[:, None] * skn + offc[None, :] * skd,
                         mask=nmask[:, None] & cmask[None, :], other=0.0)
            ksc = _bfp_scale(tl.max(tl.abs(kc), axis=1).to(tl.float32), {ib_qk})
            kI = (kc / ksc[:, None] + tl.where(kc >= 0, 0.5, -0.5)).to(tl.int32)
            kI = tl.minimum(tl.maximum(kI, {lo_qk}), {hi_qk})
{_emit_peel("k", "kI", w_qk, nD_qk, no_clamp, L2, pack_dtype)}
            cacc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
{_emit_dots("cacc", plan_qk, "q", "k", True, L2, w_qk, pack_dtype)}
            qk += cacc * qsc[:, None] * ksc[None, :]              # per-chunk block-FP, accumulated
        qk = qk * sm_scale'''

    tail = f'''        qk = tl.where(nmask[None, :], qk, -float("inf"))
        if CAUSAL:
            qk = tl.where((Q_OFF + offm // GQA_G)[:, None] >= offn[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(V + pid_z * svz + offn[:, None] * svn + offd[None, :] * svd,
                    mask=nmask[:, None] & dmask[None, :], other=0.0)
        p = p.to(tl.bfloat16)                     # P->bf16 for P@V (models the real flash datapath; NOT a packing detail, so it stays bf16 whatever pack_dtype is); l_i normalizer stays fp32
        psc = _bfp_scale(tl.max(p, axis=1).to(tl.float32), {ib_pv})
        vsc = _bfp_scale(tl.max(tl.abs(v), axis=0).to(tl.float32), {ib_pv})
        pI = (p / psc[:, None] + 0.5).to(tl.int32)
        pI = tl.minimum(pI, {hi_pv})
        vI = (v / vsc[None, :] + tl.where(v >= 0, 0.5, -0.5)).to(tl.int32)
        vI = tl.minimum(tl.maximum(vI, {lo_pv}), {hi_pv})
{_emit_peel("pp", "pI", w_pv, nD_pv, no_clamp, L, pack_dtype)}
{_emit_peel("vv", "vI", w_pv, nD_pv, no_clamp, L, pack_dtype)}
        pv = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
{_emit_dots("pv", plan_pv, "pp", "vv", False, L, w_pv, pack_dtype)}
        pv = pv * psc[:, None] * vsc[None, :]
        acc = acc * alpha[:, None] + pv
        m_i = m_new'''

    src = f'''
@triton.jit
def _flash_cg(
    Q, K, V, Out, sm_scale, Z, N_CTX, Q_LEN, Q_OFF, KVLEN,
    sqz, sqn, sqd, skz, skn, skd, svz, svn, svd, soz, son, sod,
    HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, GQA_G: tl.constexpr, KQ: tl.constexpr, HAS_KVLEN: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_z = tl.program_id(1)
    if HAS_KVLEN:                                     # per-batch valid kv length (vLLM padded decode)
        kvlen = tl.load(KVLEN + pid_z)
    else:
        kvlen = N_CTX
    offm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offd = tl.arange(0, BLOCK_D)
    dmask = offd < HEAD_DIM
    mmask = offm < Q_LEN
    qmask = mmask[:, None] & dmask[None, :]
{qk_pre}
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    if CAUSAL:
        last_m = tl.minimum((pid_m + 1) * BLOCK_M, Q_LEN) - 1
        n_end = tl.minimum(Q_OFF + last_m // GQA_G + 1, kvlen)
    else:
        n_end = kvlen
    for n0 in range(0, n_end, BLOCK_N):
        offn = n0 + tl.arange(0, BLOCK_N)
        nmask = offn < kvlen
{qk_body}
{tail}
    acc = acc / l_i[:, None]
    tl.store(Out + pid_z * soz + offm[:, None] * son + offd[None, :] * sod,
             acc.to(Out.dtype.element_ty), mask=qmask)
'''
    return src


_KERNEL_CACHE = {}


def _get_kernel(nmp_qk, w_qk, nmp_pv, w_pv, no_clamp, chunked, pack_dtype="bf16"):
    key = (nmp_qk, w_qk, nmp_pv, w_pv, no_clamp, chunked, pack_dtype)
    if key not in _KERNEL_CACHE:
        src = _gen_src(*key)
        fname = f"<flash_cg_{key}>"
        # @triton.jit reads the fn source via inspect/linecache; exec'd code has no file, so register
        # the generated source in linecache under the compile filename.
        linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
        ns = {"triton": triton, "tl": tl, "_bfp_scale": _bfp_scale}
        exec(compile(src, fname, "exec"), ns)
        _KERNEL_CACHE[key] = ns["_flash_cg"]
    return _KERNEL_CACHE[key]


@triton.jit
def _flash_exact_fwd(
    Q, K, V, Out, sm_scale, Z, N_CTX, Q_LEN, Q_OFF, KVLEN,
    sqz, sqn, sqd, skz, skn, skd, svz, svn, svd, soz, son, sod,
    HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, GQA_G: tl.constexpr, KQ: tl.constexpr, HAS_KVLEN: tl.constexpr,
    PV_SPLIT: tl.constexpr = False,
):
    """Plain bf16 flash attention (no ozaki) -- the flash-exact baseline the benches compare against.
    Same online-softmax / causal / GQA-fold structure as the generated ozaki kernel, so the two are
    apples-to-apples; only the QK/PV dots differ (single bf16 dot vs digit-plane plan).

    PV_SPLIT: carry P into P@V at ~16 mantissa bits via a hi/lo bf16 pair (two dots) instead of one
    bf16 dot. The single cast is this path's DOMINANT error term (measured 1.52e-3 of a 2.24e-3 total
    at decode N=2048, vs 2.7e-7 for everything the tiling/online-softmax does). vllm-flash-attn casts
    P to bf16 too (a mantissa-bit sweep puts it in the same 8-bit-P class), so dropping the cast takes
    this path OUT of that class rather than merely matching it: decode error falls to 1.640e-3, the
    floor set by the bf16 output store alone. Enabled for decode only (small GQA-folded BLOCK_M),
    where the extra dot rides along in a memory-bound kernel."""
    pid_m = tl.program_id(0); pid_z = tl.program_id(1)
    if HAS_KVLEN:                                     # per-batch valid kv length (vLLM padded decode)
        kvlen = tl.load(KVLEN + pid_z)
    else:
        kvlen = N_CTX
    offm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offd = tl.arange(0, BLOCK_D)
    dmask = offd < HEAD_DIM
    qmask = (offm[:, None] < Q_LEN) & dmask[None, :]
    q = tl.load(Q + pid_z * sqz + offm[:, None] * sqn + offd[None, :] * sqd, mask=qmask, other=0.0)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    if CAUSAL:
        last_m = tl.minimum((pid_m + 1) * BLOCK_M, Q_LEN) - 1
        n_end = tl.minimum(Q_OFF + last_m // GQA_G + 1, kvlen)
    else:
        n_end = kvlen
    for n0 in range(0, n_end, BLOCK_N):
        offn = n0 + tl.arange(0, BLOCK_N)
        nmask = offn < kvlen
        k = tl.load(K + pid_z * skz + offn[:, None] * skn + offd[None, :] * skd,
                    mask=nmask[:, None] & dmask[None, :], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(nmask[None, :], qk, -float("inf"))
        if CAUSAL:
            qk = tl.where((Q_OFF + offm // GQA_G)[:, None] >= offn[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(V + pid_z * svz + offn[:, None] * svn + offd[None, :] * svd,
                    mask=nmask[:, None] & dmask[None, :], other=0.0)
        if PV_SPLIT:
            # p_hi + p_lo reproduces p to ~2^-17 (p_hi is within a factor 2 of p, so the residual is
            # exact in fp32 and lands in bf16's 8 bits again). Both dots accumulate in fp32.
            p_hi = p.to(v.dtype)
            p_lo = (p - p_hi.to(tl.float32)).to(v.dtype)
            acc = acc * alpha[:, None] + tl.dot(p_hi, v) + tl.dot(p_lo, v)
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(Out + pid_z * soz + offm[:, None] * son + offd[None, :] * sod,
             acc.to(Out.dtype.element_ty), mask=qmask)


def flash_oz1fp_cg(q, k, v, nmp, w, nmp_pv=None, w_pv=None, causal=True, sm_scale=None, ozaki=True,
                   byte_split_style="all_signed_no_clamp", chunk_size=None, BLOCK_M=64, BLOCK_N=64,
                   num_warps=4, num_stages=1, kv_lens=None, pv_split=None, pack_dtype="bf16"):
    """Code-generated Flash-Ozaki1_fp. q:[B,Hq,T,D]; k,v:[B,Hkv,N,D] bf16 (MHA Hq==Hkv; GQA Hq=Hkv*G
    folds the G group-heads into the query-row dim). nmp/w drive QK^T; nmp_pv/w_pv (default = nmp/w)
    drive P@V INDEPENDENTLY -- QK-nmp != PV-nmp is supported. Optimal pack plan + single-peel planes
    are baked into the kernel source per (nmp_qk,w_qk,nmp_pv,w_pv,style). ozaki=False runs the plain
    bf16 flash path (same softmax/GQA machinery) -- the flash-exact baseline.

    chunk_size: the block-FP chunk applied to EVERY GEMM reduction, matching production. QK's head_dim
    reduction is split into KQ=next_pow2(min(chunk_size,D))-wide chunks, and PV's kv reduction chunk is
    the kv tile, so BLOCK_N is set to next_pow2(chunk_size). chunk_size=None keeps the fast un-chunked
    path (QK: one block-FP scale over the full head_dim; PV: chunk = BLOCK_N). Best power-of-2.

    kv_lens: optional int tensor [B] of the valid kv length per batch element. When given, kv positions
    >= kv_lens[b] are masked out (score -inf) -- for vLLM decode where K/V are gathered/padded to a
    shared max length but each sequence attends only its own prefix. None => all N positions valid.

    pv_split (ozaki=False only): carry P into P@V as a hi/lo bf16 pair (~16 mantissa bits, two dots)
    instead of one bf16 cast. **Default OFF** -- a mantissa-bit sweep showed vllm-flash-attn casts P
    to bf16 as well, so enabling this would make the bf16 control structurally MORE accurate (1.64e-3
    vs FA's 2.16e-3) than the kernel it is meant to stand in for. Keep it for studying the P term in
    isolation. The ozaki path never splits either -- its digit-plane PV is bit-exact with production
    `ozaki1_batched_gemm_fp` and must stay so."""
    nmp_pv = nmp if nmp_pv is None else nmp_pv
    w_pv = w if w_pv is None else w_pv
    no_clamp = 1 if byte_split_style == "all_signed_no_clamp" else 0
    B, Hq, T, D = q.shape
    Hkv, N = k.shape[1], k.shape[2]
    G = Hq // Hkv
    assert Hq == Hkv * G, f"Hq={Hq} not a multiple of Hkv={Hkv}"
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    # chunk_size drives ALL reductions: QK head_dim -> KQ chunks; PV kv -> the kv tile (BLOCK_N).
    if chunk_size is None:
        KQ, chunked = triton.next_power_of_2(D), False
    else:
        KQ = triton.next_power_of_2(min(chunk_size, D))
        chunked = KQ < D
        BLOCK_N = triton.next_power_of_2(chunk_size)          # PV reduction chunk = kv tile = chunk_size
    Qrows = T * G
    qz = (q.reshape(B * Hkv, T, D) if G == 1 else
          q.reshape(B, Hkv, G, T, D).permute(0, 1, 3, 2, 4).reshape(B * Hkv, Qrows, D)).contiguous()
    kz, vz = (t.reshape(B * Hkv, N, D).contiguous() for t in (k, v))
    BLOCK_M = max(16, min(BLOCK_M, triton.next_power_of_2(Qrows)))
    o = torch.empty_like(qz)
    kern = (_get_kernel(nmp, w, nmp_pv, w_pv, no_clamp, chunked, pack_dtype) if ozaki
            else _flash_exact_fwd)
    has_kvlen = kv_lens is not None
    # expand per-batch [B] valid-length to [B*Hkv] so program pid_z (=(b,hkv)) indexes it directly.
    kvlen_z = (kv_lens.to(device=q.device, dtype=torch.int32).reshape(B).repeat_interleave(Hkv).contiguous()
               if has_kvlen else qz)          # dummy ptr when absent; HAS_KVLEN=False so never loaded

    # OFF by default: vllm-flash-attn casts P to bf16 too, so splitting P would make this control
    # structurally MORE accurate than the kernel it stands in for. Opt in only to study the P term.
    pv_split = False if pv_split is None else pv_split
    extra = {} if ozaki else {"PV_SPLIT": bool(pv_split)}

    def _launch(bm):
        kern[(triton.cdiv(Qrows, bm), B * Hkv)](
            qz, kz, vz, o, sm_scale, B * Hkv, N, Qrows, N - T, kvlen_z,
            qz.stride(0), qz.stride(1), qz.stride(2), kz.stride(0), kz.stride(1), kz.stride(2),
            vz.stride(0), vz.stride(1), vz.stride(2), o.stride(0), o.stride(1), o.stride(2),
            HEAD_DIM=D, BLOCK_D=triton.next_power_of_2(D), BLOCK_M=bm, BLOCK_N=BLOCK_N,
            CAUSAL=causal, GQA_G=G, KQ=KQ, HAS_KVLEN=has_kvlen, num_warps=num_warps, num_stages=num_stages,
            **extra,
        )
    _run_with_oom_retry(_launch, BLOCK_M)
    if G == 1:
        return o.reshape(B, Hq, T, D)
    return o.reshape(B, Hkv, T, G, D).permute(0, 1, 3, 2, 4).reshape(B, Hq, T, D)


# --- pre-encoded KV cache (the attention analog of weight_cache) -------------------------------
def _gen_cached_src(nmp_qk, w_qk, nmp_pv, w_pv, no_clamp, chunked, pack_dtype="bf16"):
    """Codegen cached-KV flash kernel: same plan+peel as _gen_src, but K/V are LOADED from
    pre-encoded place-folded planes (encode_kv) instead of peeled inline -- so per-tile K/V amax +
    round + split + cast is skipped (decode's dominant cost). Q & P are still encoded inline (P is the
    live softmax, uncacheable). K planes -> kp{t}, V planes -> vvp{t}, matching _emit_dots' _super().

    chunked mirrors _gen_src: hoisted (one head_dim K scale, Q peeled once) vs chunked (K head_dim
    reduction split into KQ chunks with per-chunk cached K scales from k_scale[Z,nchd,N]; Q re-encoded
    per kv tile). PV always chunks by the kv tile (BLOCK_N)."""
    nD_qk = oz1fp_params(nmp_qk, w_qk)[0]
    nD_pv = oz1fp_params(nmp_pv, w_pv)[0]
    ib_qk = w_qk * nD_qk - 1
    ib_pv = w_pv * nD_pv - 1
    plan_qk = pack_plan(nmp_qk, w_qk, pack_dtype)
    plan_pv = pack_plan(nmp_pv, w_pv, pack_dtype)
    I, L, L2 = "    ", "        ", "            "
    lo_qk, hi_qk, hi_pv, lo_pv = -(1 << ib_qk), (1 << ib_qk) - 1, (1 << ib_pv) - 1, -(1 << ib_pv)
    # load names MUST match _super(prefix): _super("k")->kp{t}, _super("vv")->vvp{t} (it appends 'p').
    vload = "\n".join(
        f"{L}vvp{t} = tl.load(Vp + pid_z*svz + {t}*svt + offn[:,None]*svn + offd[None,:]*svd,"
        f" mask=nmask[:,None] & dmask[None,:], other=0.0)" for t in range(nD_pv))
    if not chunked:
        kload = "\n".join(
            f"{L}kp{t} = tl.load(Kp + pid_z*skz + {t}*skt + offn[:,None]*skn + offd[None,:]*skd,"
            f" mask=nmask[:,None] & dmask[None,:], other=0.0)" for t in range(nD_qk))
        qk_pre = f'''    q = tl.load(Q + pid_z * sqz + offm[:, None] * sqn + offd[None, :] * sqd, mask=qmask, other=0.0)
    qsc = _bfp_scale(tl.max(tl.abs(q), axis=1).to(tl.float32), {ib_qk})
    qI = (q / qsc[:, None] + tl.where(q >= 0, 0.5, -0.5)).to(tl.int32)
    qI = tl.minimum(tl.maximum(qI, {lo_qk}), {hi_qk})
{_emit_peel("q", "qI", w_qk, nD_qk, no_clamp, I, pack_dtype)}'''
        qk_body = f'''        ksc = tl.load(Ks + pid_z * sksz + offn * sksn, mask=nmask, other=0.0)
{kload}
        cacc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
{_emit_dots("cacc", plan_qk, "q", "k", True, L, w_qk, pack_dtype)}
        qk = cacc * qsc[:, None] * ksc[None, :] * sm_scale'''
    else:
        kload = "\n".join(
            f"{L2}kp{t} = tl.load(Kp + pid_z*skz + {t}*skt + offn[:,None]*skn + offc[None,:]*skd,"
            f" mask=nmask[:,None] & cmask[None,:], other=0.0)" for t in range(nD_qk))
        qk_pre = ""
        qk_body = f'''        qk = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for dc in range(0, HEAD_DIM, KQ):                         # chunk the head_dim reduction by KQ
            offc = dc + tl.arange(0, KQ)
            cmask = offc < HEAD_DIM
            qc = tl.load(Q + pid_z * sqz + offm[:, None] * sqn + offc[None, :] * sqd,
                         mask=mmask[:, None] & cmask[None, :], other=0.0)
            qsc = _bfp_scale(tl.max(tl.abs(qc), axis=1).to(tl.float32), {ib_qk})
            qI = (qc / qsc[:, None] + tl.where(qc >= 0, 0.5, -0.5)).to(tl.int32)
            qI = tl.minimum(tl.maximum(qI, {lo_qk}), {hi_qk})
{_emit_peel("q", "qI", w_qk, nD_qk, no_clamp, L2, pack_dtype)}
            ksc = tl.load(Ks + pid_z * sksz + (dc // KQ) * skscd + offn * sksn, mask=nmask, other=0.0)
{kload}
            cacc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
{_emit_dots("cacc", plan_qk, "q", "k", True, L2, w_qk, pack_dtype)}
            qk += cacc * qsc[:, None] * ksc[None, :]              # per-chunk cached K scale, accumulated
        qk = qk * sm_scale'''

    tail = f'''        qk = tl.where(nmask[None, :], qk, -float("inf"))
        if CAUSAL:
            qk = tl.where((Q_OFF + offm // GQA_G)[:, None] >= offn[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        vsc = tl.load(Vs + pid_z * svsz + (n0 // BLOCK_N) * svsc + offd * svsd, mask=dmask, other=0.0)
        p = p.to(tl.bfloat16)                     # P->bf16 for P@V (matches SDPA/flash-exact); l_i normalizer stays fp32
        psc = _bfp_scale(tl.max(p, axis=1).to(tl.float32), {ib_pv})
        pI = (p / psc[:, None] + 0.5).to(tl.int32)
        pI = tl.minimum(pI, {hi_pv})
{_emit_peel("pp", "pI", w_pv, nD_pv, no_clamp, L, pack_dtype)}
{vload}
        pv = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
{_emit_dots("pv", plan_pv, "pp", "vv", False, L, w_pv, pack_dtype)}
        pv = pv * psc[:, None] * vsc[None, :]
        acc = acc * alpha[:, None] + pv
        m_i = m_new'''

    src = f'''
@triton.jit
def _flash_cg_cached(
    Q, Kp, Ks, Vp, Vs, Out, sm_scale, Z, N_CTX, Q_LEN, Q_OFF, KVLEN,
    sqz, sqn, sqd, skz, skt, skn, skd, sksz, skscd, sksn,
    svz, svt, svn, svd, svsz, svsc, svsd, soz, son, sod,
    HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, GQA_G: tl.constexpr, KQ: tl.constexpr, HAS_KVLEN: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_z = tl.program_id(1)
    if HAS_KVLEN:                                     # per-batch valid kv length (vLLM padded decode)
        kvlen = tl.load(KVLEN + pid_z)
    else:
        kvlen = N_CTX
    offm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offd = tl.arange(0, BLOCK_D)
    dmask = offd < HEAD_DIM
    mmask = offm < Q_LEN
    qmask = mmask[:, None] & dmask[None, :]
{qk_pre}
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    if CAUSAL:
        last_m = tl.minimum((pid_m + 1) * BLOCK_M, Q_LEN) - 1
        n_end = tl.minimum(Q_OFF + last_m // GQA_G + 1, kvlen)
    else:
        n_end = kvlen
    for n0 in range(0, n_end, BLOCK_N):
        offn = n0 + tl.arange(0, BLOCK_N)
        nmask = offn < kvlen
{qk_body}
{tail}
    acc = acc / l_i[:, None]
    tl.store(Out + pid_z * soz + offm[:, None] * son + offd[None, :] * sod,
             acc.to(Out.dtype.element_ty), mask=qmask)
'''
    return src


_CACHED_KERNEL_CACHE = {}


def _get_cached_kernel(nmp_qk, w_qk, nmp_pv, w_pv, no_clamp, chunked):
    key = (nmp_qk, w_qk, nmp_pv, w_pv, no_clamp, chunked)
    if key not in _CACHED_KERNEL_CACHE:
        src = _gen_cached_src(*key)
        fname = f"<flash_cg_cached_{key}>"
        linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
        ns = {"triton": triton, "tl": tl, "_bfp_scale": _bfp_scale}
        exec(compile(src, fname, "exec"), ns)
        _CACHED_KERNEL_CACHE[key] = ns["_flash_cg_cached"]
    return _CACHED_KERNEL_CACHE[key]


def encode_kv(k, v, nmp, w, nmp_pv=None, w_pv=None, byte_split_style="all_signed_no_clamp",
              chunk_size=None, block_n=64):
    """k,v:[B,H,N,D] bf16 -> (k_pl[Z,nD_qk,N,D], k_scale[Z,nchd,N], v_pl[Z,nD_pv,N,D], v_scale[Z,nch,D]),
    Z=B*H. Production-faithful block-FP (_bfp_scale_torch: frexp int_bits, clamp). Planes are
    place-folded and encoded exactly like the kernel's inline peel, so cached == non-cached.

    chunk_size sets the block-FP chunk for BOTH cached reductions (must match
    flash_oz1fp_cg_cached(chunk_size=...)): K's head_dim is scaled per KQ=next_pow2(min(chunk_size,D))
    chunk (nchd=ceil(D/KQ) scales per token -> k_scale[Z,nchd,N]) and V's kv per next_pow2(chunk_size)
    chunk. chunk_size=None -> one head_dim scale (nchd=1, k_scale[Z,1,N]) and V per `block_n`.
    QK uses (nmp,w); PV uses (nmp_pv,w_pv), default (nmp,w)."""
    nmp_pv = nmp if nmp_pv is None else nmp_pv
    w_pv = w if w_pv is None else w_pv
    no_clamp = byte_split_style == "all_signed_no_clamp"
    B, H, N, D = k.shape
    Z = B * H
    nD_qk = oz1fp_params(nmp, w)[0]; ib_qk = w * nD_qk - 1
    nD_pv = oz1fp_params(nmp_pv, w_pv)[0]; ib_pv = w_pv * nD_pv - 1
    if chunk_size is not None:
        block_n = triton.next_power_of_2(chunk_size)                            # V kv chunk = kernel tile
    KQ = D if chunk_size is None else min(triton.next_power_of_2(chunk_size), D)  # K head_dim chunk
    nchd = (D + KQ - 1) // KQ
    rnd = lambda x: torch.trunc(x + torch.where(x >= 0, 0.5, -0.5))   # round-half-away, matches the kernel
    kf = k.reshape(Z, N, D).float(); vf = v.reshape(Z, N, D).float()
    k_scale = torch.empty(Z, nchd, N, device=k.device, dtype=torch.float32)     # per (head_dim-chunk, token)
    kI = torch.empty(Z, N, D, device=k.device, dtype=torch.float32)
    for c in range(nchd):
        s, e = c * KQ, min((c + 1) * KQ, D)
        sc = _bfp_scale_torch(kf[:, :, s:e].abs().amax(-1, keepdim=True), ib_qk)  # [Z,N,1]
        k_scale[:, c, :] = sc[:, :, 0]
        kI[:, :, s:e] = rnd(kf[:, :, s:e] / sc).clamp(-(1 << ib_qk), (1 << ib_qk) - 1)
    k_pl = _digit_planes(kI.to(torch.int64), w, nD_qk, no_clamp)                # [Z,nD_qk,N,D]
    nch = (N + block_n - 1) // block_n                                          # V chunked = kernel tiles
    v_scale = torch.empty(Z, nch, D, device=v.device, dtype=torch.float32)
    XiV = torch.empty(Z, N, D, device=v.device, dtype=torch.float32)
    for c in range(nch):
        s, e = c * block_n, min((c + 1) * block_n, N)
        sc = _bfp_scale_torch(vf[:, s:e, :].abs().amax(1, keepdim=True), ib_pv)   # [Z,1,D] per (chunk,dim)
        v_scale[:, c, :] = sc[:, 0, :]
        XiV[:, s:e, :] = rnd(vf[:, s:e, :] / sc).clamp(-(1 << ib_pv), (1 << ib_pv) - 1)
    v_pl = _digit_planes(XiV.to(torch.int64), w_pv, nD_pv, no_clamp)            # [Z,nD_pv,N,D]
    return (k_pl.contiguous(), k_scale.contiguous(),
            v_pl.contiguous(), v_scale.contiguous())


def encode_kv_append(kv, k_new, v_new, nmp, w, nmp_pv=None, w_pv=None,
                     byte_split_style="all_signed_no_clamp", chunk_size=None, block_n=64):
    """INCREMENTAL decode append: encode ONLY the new tokens' K/V (k_new,v_new:[B,H,n_new,D]) and
    concatenate to an existing cache `kv` along the kv-length axis. Cost is O(n_new), independent of
    the cached length -- vs re-encoding the whole cache every step. Because V is block-FP-scaled per
    `block_n`-token chunk, n_new must be a multiple of the V chunk (next_pow2(chunk_size) if chunk_size
    else block_n) so the appended v_scale rows align to whole chunks. Returns the extended cache."""
    k_pl, k_scale, v_pl, v_scale = kv
    n = encode_kv(k_new, v_new, nmp, w, nmp_pv, w_pv, byte_split_style, chunk_size, block_n)
    bn = triton.next_power_of_2(chunk_size) if chunk_size is not None else block_n
    assert k_new.shape[2] % bn == 0, \
        f"n_new={k_new.shape[2]} must be a multiple of the V chunk {bn} to keep v_scale chunk-aligned"
    return (torch.cat([k_pl, n[0]], dim=2),        # [Z,nD_qk,N,D] on N
            torch.cat([k_scale, n[1]], dim=2),     # [Z,nchd,N]    on N
            torch.cat([v_pl, n[2]], dim=2),        # [Z,nD_pv,N,D] on N
            torch.cat([v_scale, n[3]], dim=1))     # [Z,nch,D]     on chunk axis


# --- OPTIONAL: fused-triton incremental encoder (fast decode append, no torch encode / no concat) ---
@triton.jit
def _enc_k_kernel(K, Kp, Ks, OFF, N_NEW,
                  skz, skn, skd, spz, spt, spn, spd, ssz, ssc, ssn,
                  D: tl.constexpr, IB: tl.constexpr, W: tl.constexpr, ND: tl.constexpr,
                  NO_CLAMP: tl.constexpr, KQ: tl.constexpr, BLK: tl.constexpr):
    """Encode BLK new K tokens x one head_dim chunk (KQ cols) -> place-folded planes + per-token scale,
    written in-place at kv position OFF. K scale is per-token over the KQ head_dim chunk (matches
    encode_kv / the cached kernel)."""
    pt = tl.program_id(0); pc = tl.program_id(1); pz = tl.program_id(2)
    offn = pt * BLK + tl.arange(0, BLK); nmask = offn < N_NEW
    offd = pc * KQ + tl.arange(0, KQ); dmask = offd < D
    m = nmask[:, None] & dmask[None, :]
    k = tl.load(K + pz * skz + offn[:, None] * skn + offd[None, :] * skd, mask=m, other=0.0)
    sc = _bfp_scale(tl.max(tl.abs(k), axis=1).to(tl.float32), IB)
    kI = (k / sc[:, None] + tl.where(k >= 0, 0.5, -0.5)).to(tl.int32)
    kI = tl.minimum(tl.maximum(kI, -(1 << IB)), (1 << IB) - 1)
    tl.store(Ks + pz * ssz + pc * ssc + (OFF + offn) * ssn, sc, mask=nmask)
    cur = kI
    for t in range(ND):                                          # ND constexpr -> unrolled
        if t == ND - 1 and NO_CLAMP != 0:
            d = cur
        else:
            lo = cur & ((1 << W) - 1)
            d = tl.where(lo >= (1 << (W - 1)), lo - (1 << W), lo)
        pl = (d * (1 << (W * t))).to(tl.bfloat16)
        tl.store(Kp + pz * spz + t * spt + (OFF + offn)[:, None] * spn + offd[None, :] * spd, pl, mask=m)
        if t != ND - 1:
            cur = (cur - d) >> W


@triton.jit
def _enc_v_kernel(V, Vp, Vs, OFF, N_NEW,
                  svz, svn, svd, spz, spt, spn, spd, ssz, ssc, ssd,
                  D: tl.constexpr, IB: tl.constexpr, W: tl.constexpr, ND: tl.constexpr,
                  NO_CLAMP: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    """Encode one BN-token V chunk x BD dims -> place-folded planes + per-(chunk,dim) scale, in-place
    at OFF. V scale is per-dim over the BN kv tokens (axis 0), matching encode_kv / the cached kernel."""
    pc = tl.program_id(0); pd = tl.program_id(1); pz = tl.program_id(2)
    offn = pc * BN + tl.arange(0, BN); nmask = offn < N_NEW
    offd = pd * BD + tl.arange(0, BD); dmask = offd < D
    m = nmask[:, None] & dmask[None, :]
    v = tl.load(V + pz * svz + offn[:, None] * svn + offd[None, :] * svd, mask=m, other=0.0)
    sc = _bfp_scale(tl.max(tl.abs(v), axis=0).to(tl.float32), IB)
    vI = (v / sc[None, :] + tl.where(v >= 0, 0.5, -0.5)).to(tl.int32)
    vI = tl.minimum(tl.maximum(vI, -(1 << IB)), (1 << IB) - 1)
    tl.store(Vs + pz * ssz + (OFF // BN + pc) * ssc + offd * ssd, sc, mask=dmask)
    cur = vI
    for t in range(ND):
        if t == ND - 1 and NO_CLAMP != 0:
            d = cur
        else:
            lo = cur & ((1 << W) - 1)
            d = tl.where(lo >= (1 << (W - 1)), lo - (1 << W), lo)
        pl = (d * (1 << (W * t))).to(tl.bfloat16)
        tl.store(Vp + pz * spz + t * spt + (OFF + offn)[:, None] * spn + offd[None, :] * spd, pl, mask=m)
        if t != ND - 1:
            cur = (cur - d) >> W


def alloc_kv_cache(B, H, N_max, D, nmp, w, nmp_pv=None, w_pv=None, chunk_size=None, block_n=64,
                   device="cuda"):
    """Pre-allocate a zeroed KV-plane cache to N_max tokens (for encode_kv_append_fused in-place writes)."""
    nmp_pv = nmp if nmp_pv is None else nmp_pv
    w_pv = w if w_pv is None else w_pv
    Z = B * H
    nD_qk = oz1fp_params(nmp, w)[0]; nD_pv = oz1fp_params(nmp_pv, w_pv)[0]
    if chunk_size is not None:
        block_n = triton.next_power_of_2(chunk_size)
    KQ = D if chunk_size is None else min(triton.next_power_of_2(chunk_size), D)
    nchd = (D + KQ - 1) // KQ
    nch = (N_max + block_n - 1) // block_n
    return [torch.zeros(Z, nD_qk, N_max, D, device=device, dtype=torch.bfloat16),
            torch.zeros(Z, nchd, N_max, device=device, dtype=torch.float32),
            torch.zeros(Z, nD_pv, N_max, D, device=device, dtype=torch.bfloat16),
            torch.zeros(Z, nch, D, device=device, dtype=torch.float32)]


def encode_kv_append_fused(cache, off, k_new, v_new, nmp, w, nmp_pv=None, w_pv=None,
                           byte_split_style="all_signed_no_clamp", chunk_size=None, block_n=64):
    """FUSED-triton incremental encode: write new tokens' K/V planes+scales **in-place** into the
    pre-allocated `cache` (alloc_kv_cache) at kv position `off` -- no torch encode_kv, no O(N) concat,
    so per-append cost is O(n_new) fused-triton (~tens of us). n_new and off must be block_n-aligned.
    Bit-identical to encode_kv over the same tokens. Returns off + n_new (the new cache length)."""
    nmp_pv = nmp if nmp_pv is None else nmp_pv
    w_pv = w if w_pv is None else w_pv
    no_clamp = 1 if byte_split_style == "all_signed_no_clamp" else 0
    B, H, n_new, D = k_new.shape
    Z = B * H
    nD_qk = oz1fp_params(nmp, w)[0]; ib_qk = w * nD_qk - 1
    nD_pv = oz1fp_params(nmp_pv, w_pv)[0]; ib_pv = w_pv * nD_pv - 1
    if chunk_size is not None:
        block_n = triton.next_power_of_2(chunk_size)
    KQ = D if chunk_size is None else min(triton.next_power_of_2(chunk_size), D)
    nchd = (D + KQ - 1) // KQ
    assert n_new % block_n == 0 and off % block_n == 0, \
        f"n_new={n_new} and off={off} must be multiples of block_n={block_n} (V-chunk alignment)"
    k_pl, k_scale, v_pl, v_scale = cache
    kz = k_new.reshape(Z, n_new, D).contiguous(); vz = v_new.reshape(Z, n_new, D).contiguous()
    BLK = 32
    _enc_k_kernel[(triton.cdiv(n_new, BLK), nchd, Z)](
        kz, k_pl, k_scale, off, n_new,
        kz.stride(0), kz.stride(1), kz.stride(2),
        k_pl.stride(0), k_pl.stride(1), k_pl.stride(2), k_pl.stride(3),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        D=D, IB=ib_qk, W=w, ND=nD_qk, NO_CLAMP=no_clamp, KQ=KQ, BLK=BLK, num_warps=4)
    BD = triton.next_power_of_2(D)
    _enc_v_kernel[(triton.cdiv(n_new, block_n), triton.cdiv(D, BD), Z)](
        vz, v_pl, v_scale, off, n_new,
        vz.stride(0), vz.stride(1), vz.stride(2),
        v_pl.stride(0), v_pl.stride(1), v_pl.stride(2), v_pl.stride(3),
        v_scale.stride(0), v_scale.stride(1), v_scale.stride(2),
        D=D, IB=ib_pv, W=w_pv, ND=nD_pv, NO_CLAMP=no_clamp, BN=block_n, BD=BD, num_warps=4)
    return off + n_new


def flash_oz1fp_cg_cached(q, kv, nmp, w, nmp_pv=None, w_pv=None, causal=True, sm_scale=None,
                          byte_split_style="all_signed_no_clamp", chunk_size=None, BLOCK_M=64, BLOCK_N=64,
                          num_warps=4, num_stages=1, kv_lens=None):
    """Cached-KV codegen flash. q:[B,Hq,T,D]; kv = encode_kv(..., chunk_size=chunk_size) (Hkv kv heads,
    length N). MHA: Hq==Hkv. GQA: Hq=Hkv*G -> the G query heads sharing each cached kv head fold into
    the query-row dim, so one cache tile feeds all G. Prefill T==N; decode/chunked T<N (q token i at abs
    kv pos N-T+i). EXACT vs non-cached flash_oz1fp_cg for the same config. chunk_size MUST match the
    encode_kv call: it chunks K's head_dim (KQ, via k_scale[Z,nchd,N]) and V's kv (BLOCK_N).

    OPT-IN, not the serving default: the cache stores nD digit planes for K and V, so KV-cache memory
    grows nD x (3-5x for w4 nmp9-15) for only a ~1.3x speedup (RESULTS.md E.2/E.4). Since vLLM throughput
    is bound by KV-cache capacity (concurrency x context), that nD x blowup is a net throughput loss --
    prefer non-cached flash_oz1fp_cg (re-encodes K/V in-kernel, KV stays 1x). Use this cached path only
    for low-concurrency / latency-critical single-sequence cases where KV memory is not the bottleneck."""
    nmp_pv = nmp if nmp_pv is None else nmp_pv
    w_pv = w if w_pv is None else w_pv
    no_clamp = 1 if byte_split_style == "all_signed_no_clamp" else 0
    B, Hq, T, D = q.shape
    k_pl, k_scale, v_pl, v_scale = kv
    Zc, nchd, N = k_scale.shape                                # cache: Zc=B*Hkv, nchd head_dim chunks, N kv
    Hkv = Zc // B
    G = Hq // Hkv
    assert Hq == Hkv * G, f"Hq={Hq} not a multiple of cache Hkv={Hkv}"
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    # chunk_size drives both reductions and MUST match how the cache was encoded (nchd, V chunk).
    if chunk_size is None:
        KQ, chunked = D, False
    else:
        KQ = min(triton.next_power_of_2(chunk_size), D)
        chunked = KQ < D
        BLOCK_N = triton.next_power_of_2(chunk_size)
    assert nchd == triton.cdiv(D, KQ), \
        f"cache has {nchd} head_dim chunks but chunk_size implies {triton.cdiv(D, KQ)}; pass the same chunk_size to encode_kv"
    assert v_scale.shape[1] == triton.cdiv(N, BLOCK_N), \
        f"V cache chunked at {N // v_scale.shape[1]} but BLOCK_N={BLOCK_N}; encode_kv(block_n=)/chunk_size must match"
    Qrows = T * G
    qz = (q.reshape(Zc, T, D) if G == 1 else
          q.reshape(B, Hkv, G, T, D).permute(0, 1, 3, 2, 4).reshape(Zc, Qrows, D)).contiguous()
    BLOCK_M = max(16, min(BLOCK_M, triton.next_power_of_2(Qrows)))
    o = torch.empty_like(qz)
    kern = _get_cached_kernel(nmp, w, nmp_pv, w_pv, no_clamp, chunked)
    has_kvlen = kv_lens is not None
    kvlen_z = (kv_lens.to(device=q.device, dtype=torch.int32).reshape(B).repeat_interleave(Hkv).contiguous()
               if has_kvlen else qz)          # dummy ptr when absent; HAS_KVLEN=False so never loaded

    def _launch(bm):
        kern[(triton.cdiv(Qrows, bm), Zc)](
            qz, k_pl, k_scale, v_pl, v_scale, o, sm_scale, Zc, N, Qrows, N - T, kvlen_z,
            qz.stride(0), qz.stride(1), qz.stride(2),
            k_pl.stride(0), k_pl.stride(1), k_pl.stride(2), k_pl.stride(3),
            k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
            v_pl.stride(0), v_pl.stride(1), v_pl.stride(2), v_pl.stride(3),
            v_scale.stride(0), v_scale.stride(1), v_scale.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            HEAD_DIM=D, BLOCK_D=triton.next_power_of_2(D), BLOCK_M=bm, BLOCK_N=BLOCK_N,
            CAUSAL=causal, GQA_G=G, KQ=KQ, HAS_KVLEN=has_kvlen, num_warps=num_warps, num_stages=num_stages,
        )
    _run_with_oom_retry(_launch, BLOCK_M)
    if G == 1:
        return o.reshape(B, Hq, T, D)
    return o.reshape(B, Hkv, T, G, D).permute(0, 1, 3, 2, 4).reshape(B, Hq, T, D)


# --- split-KV / flash-decoding (opt-in, decode) -----------------------------------------------
# ncu shows ozaki DECODE is integer-ALU-bound (~44%) at only ~13% occupancy with DRAM ~30% idle
# (exact decode is DRAM-bound at ~96%): the grid B*Hkv is too small to fill the SMs, so the ALU
# emulation can't hide behind the KV-read memory traffic. Split the kv loop into N_SPLITS program-z
# slices (grid gains an S axis), each computing a PARTIAL online-softmax (m,l,acc); a combine kernel
# merges them by log-sum-exp. More concurrent programs -> higher occupancy -> the ALU overlaps the
# idle memory pipe. chunk=32 only; each split is a whole number of BLOCK_N tiles so per-tile block-FP
# is byte-for-byte the non-split encoding (result matches non-split to fp-accumulation order ~1e-6).
def _gen_split_src(nmp_qk, w_qk, nmp_pv, w_pv, no_clamp, pack_dtype="bf16"):
    assert_int32_peel_fits(nmp_qk, w_qk); assert_int32_peel_fits(nmp_pv, w_pv)
    nD_qk = oz1fp_params(nmp_qk, w_qk)[0]
    nD_pv = oz1fp_params(nmp_pv, w_pv)[0]
    ib_qk = w_qk * nD_qk - 1
    ib_pv = w_pv * nD_pv - 1
    plan_qk = pack_plan(nmp_qk, w_qk, pack_dtype)
    plan_pv = pack_plan(nmp_pv, w_pv, pack_dtype)
    L, L2 = "        ", "            "
    lo_qk, hi_qk = -(1 << ib_qk), (1 << ib_qk) - 1
    lo_pv, hi_pv = -(1 << ib_pv), (1 << ib_pv) - 1
    qk_body = f'''        qk = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for dc in range(0, HEAD_DIM, KQ):
            offc = dc + tl.arange(0, KQ)
            cmask = offc < HEAD_DIM
            qc = tl.load(Q + pid_z * sqz + offm[:, None] * sqn + offc[None, :] * sqd,
                         mask=mmask[:, None] & cmask[None, :], other=0.0)
            qsc = _bfp_scale(tl.max(tl.abs(qc), axis=1).to(tl.float32), {ib_qk})
            qI = (qc / qsc[:, None] + tl.where(qc >= 0, 0.5, -0.5)).to(tl.int32)
            qI = tl.minimum(tl.maximum(qI, {lo_qk}), {hi_qk})
{_emit_peel("q", "qI", w_qk, nD_qk, no_clamp, L2, pack_dtype)}
            kc = tl.load(K + pid_z * skz + offn[:, None] * skn + offc[None, :] * skd,
                         mask=nmask[:, None] & cmask[None, :], other=0.0)
            ksc = _bfp_scale(tl.max(tl.abs(kc), axis=1).to(tl.float32), {ib_qk})
            kI = (kc / ksc[:, None] + tl.where(kc >= 0, 0.5, -0.5)).to(tl.int32)
            kI = tl.minimum(tl.maximum(kI, {lo_qk}), {hi_qk})
{_emit_peel("k", "kI", w_qk, nD_qk, no_clamp, L2, pack_dtype)}
            cacc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
{_emit_dots("cacc", plan_qk, "q", "k", True, L2, w_qk, pack_dtype)}
            qk += cacc * qsc[:, None] * ksc[None, :]
        qk = qk * sm_scale'''
    tail = f'''        qk = tl.where(nmask[None, :], qk, -float("inf"))
        if CAUSAL:
            qk = tl.where((Q_OFF + offm // GQA_G)[:, None] >= offn[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(V + pid_z * svz + offn[:, None] * svn + offd[None, :] * svd,
                    mask=nmask[:, None] & dmask[None, :], other=0.0)
        p = p.to(tl.bfloat16)
        psc = _bfp_scale(tl.max(p, axis=1).to(tl.float32), {ib_pv})
        vsc = _bfp_scale(tl.max(tl.abs(v), axis=0).to(tl.float32), {ib_pv})
        pI = (p / psc[:, None] + 0.5).to(tl.int32)
        pI = tl.minimum(pI, {hi_pv})
        vI = (v / vsc[None, :] + tl.where(v >= 0, 0.5, -0.5)).to(tl.int32)
        vI = tl.minimum(tl.maximum(vI, {lo_pv}), {hi_pv})
{_emit_peel("pp", "pI", w_pv, nD_pv, no_clamp, L, pack_dtype)}
{_emit_peel("vv", "vI", w_pv, nD_pv, no_clamp, L, pack_dtype)}
        pv = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
{_emit_dots("pv", plan_pv, "pp", "vv", False, L, w_pv, pack_dtype)}
        pv = pv * psc[:, None] * vsc[None, :]
        acc = acc * alpha[:, None] + pv
        m_i = m_new'''
    src = f'''
@triton.jit
def _flash_split(
    Q, K, V, Mp, Lp, Accp, sm_scale, Z, N_CTX, Q_LEN, Q_OFF, KVLEN,
    sqz, sqn, sqd, skz, skn, skd, svz, svn, svd,
    smz, sms, smm, slz, sls, slm, saz, sas, sam, sad,
    HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, GQA_G: tl.constexpr, KQ: tl.constexpr, HAS_KVLEN: tl.constexpr,
    N_SPLITS: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_z = tl.program_id(1); pid_s = tl.program_id(2)
    if HAS_KVLEN:
        kvlen = tl.load(KVLEN + pid_z)
    else:
        kvlen = N_CTX
    offm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offd = tl.arange(0, BLOCK_D)
    dmask = offd < HEAD_DIM
    mmask = offm < Q_LEN
    if CAUSAL:
        last_m = tl.minimum((pid_m + 1) * BLOCK_M, Q_LEN) - 1
        n_end = tl.minimum(Q_OFF + last_m // GQA_G + 1, kvlen)
    else:
        n_end = kvlen
    sblk = tl.cdiv(tl.cdiv(n_end, BLOCK_N), N_SPLITS)         # kv tiles per split
    kv_start = pid_s * sblk * BLOCK_N
    kv_end = tl.minimum((pid_s + 1) * sblk * BLOCK_N, n_end)  # empty (start>=end) => partial stays (-inf,0,0)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    for n0 in range(kv_start, kv_end, BLOCK_N):
        offn = n0 + tl.arange(0, BLOCK_N)
        nmask = offn < kvlen
{qk_body}
{tail}
    tl.store(Mp + pid_z * smz + pid_s * sms + offm * smm, m_i, mask=mmask)
    tl.store(Lp + pid_z * slz + pid_s * sls + offm * slm, l_i, mask=mmask)
    tl.store(Accp + pid_z * saz + pid_s * sas + offm[:, None] * sam + offd[None, :] * sad,
             acc, mask=mmask[:, None] & dmask[None, :])
'''
    return src


@triton.jit
def _flash_exact_split_fwd(
    Q, K, V, Mp, Lp, Accp, sm_scale, Z, N_CTX, Q_LEN, Q_OFF, KVLEN,
    sqz, sqn, sqd, skz, skn, skd, svz, svn, svd,
    smz, sms, smm, slz, sls, slm, saz, sas, sam, sad,
    HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, GQA_G: tl.constexpr, KQ: tl.constexpr, HAS_KVLEN: tl.constexpr,
    N_SPLITS: tl.constexpr, PV_SPLIT: tl.constexpr = False,
):
    """Plain bf16 twin of the generated `_flash_split`: flash-decoding partials (fp32 m/l/acc per kv
    slice) merged by `_flash_combine`'s fp32 LSE sum. This is vllm-flash-attn's decode structure --
    measured: FA at num_splits=1 lands on 2.234e-3 (== our single-pass path) and its auto choice of
    ~8 splits on 2.163e-3, so the split count, not the dot or the softmax, was the whole difference."""
    pid_m = tl.program_id(0); pid_z = tl.program_id(1); pid_s = tl.program_id(2)
    if HAS_KVLEN:
        kvlen = tl.load(KVLEN + pid_z)
    else:
        kvlen = N_CTX
    offm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offd = tl.arange(0, BLOCK_D)
    dmask = offd < HEAD_DIM
    mmask = offm < Q_LEN
    q = tl.load(Q + pid_z * sqz + offm[:, None] * sqn + offd[None, :] * sqd,
                mask=mmask[:, None] & dmask[None, :], other=0.0)
    if CAUSAL:
        last_m = tl.minimum((pid_m + 1) * BLOCK_M, Q_LEN) - 1
        n_end = tl.minimum(Q_OFF + last_m // GQA_G + 1, kvlen)
    else:
        n_end = kvlen
    sblk = tl.cdiv(tl.cdiv(n_end, BLOCK_N), N_SPLITS)         # kv tiles per split
    kv_start = pid_s * sblk * BLOCK_N
    kv_end = tl.minimum((pid_s + 1) * sblk * BLOCK_N, n_end)  # empty slice keeps the (-inf, 0, 0) partial
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    for n0 in range(kv_start, kv_end, BLOCK_N):
        offn = n0 + tl.arange(0, BLOCK_N)
        nmask = offn < kvlen
        k = tl.load(K + pid_z * skz + offn[:, None] * skn + offd[None, :] * skd,
                    mask=nmask[:, None] & dmask[None, :], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(nmask[None, :], qk, -float("inf"))
        if CAUSAL:
            qk = tl.where((Q_OFF + offm // GQA_G)[:, None] >= offn[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(V + pid_z * svz + offn[:, None] * svn + offd[None, :] * svd,
                    mask=nmask[:, None] & dmask[None, :], other=0.0)
        if PV_SPLIT:
            p_hi = p.to(v.dtype)
            p_lo = (p - p_hi.to(tl.float32)).to(v.dtype)
            acc = acc * alpha[:, None] + tl.dot(p_hi, v) + tl.dot(p_lo, v)
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    tl.store(Mp + pid_z * smz + pid_s * sms + offm * smm, m_i, mask=mmask)
    tl.store(Lp + pid_z * slz + pid_s * sls + offm * slm, l_i, mask=mmask)
    tl.store(Accp + pid_z * saz + pid_s * sas + offm[:, None] * sam + offd[None, :] * sad,
             acc, mask=mmask[:, None] & dmask[None, :])


def decode_num_splits(zc, n_tiles, n_sm=None):
    """THE decode split count -- used by BOTH the ozaki path and the bf16 exact control, on purpose.

    Sizing is occupancy-driven (the ozaki kernel is integer-ALU-bound at decode and its natural grid is
    only zc = B*Hkv, see Part F.5: up to 3.8x from splitting). The exact control deliberately reuses it
    rather than vllm-flash-attn's own count, because the split count is not numerically neutral: each
    split re-normalises by its own max, so its argmax lands on p = exp(0) = 1, which is exactly
    representable in bf16 (eps = 0) AND is the heaviest term in that split. More splits => more
    zero-error anchors => lower error (measured w4 nmp10: 2.267e-3 at 1 split, 2.093e-3 at 32).

    A control that split differently from the run it controls would carry a different amount of that
    artefact -- which is exactly what happened before: the ozaki runs split 32-way while the bf16
    control ran single-pass, leaving the CONTROL less accurate than the arm it was controlling.
    Sharing this function makes the two structurally identical, so an ozaki-vs-control difference is
    the digit-plane GEMM and nothing else.

    `fa_num_splits` is the vllm-flash-attn-matched alternative, kept for measuring against FA."""
    if n_sm is None:
        n_sm = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return max(1, min(-(-16 * n_sm // max(zc, 1)), n_tiles, 32))


def fa_num_splits(zc, n_tiles, n_sm=None):
    """flash-decoding split count sized to land on vllm-flash-attn's own choice. NOT the serving
    default -- see `decode_num_splits` for why both paths share the occupancy heuristic instead.
    Use this when the question is "how do we compare against FA", not "isolate ozaki".

    FA2 picks num_splits by wave-quantisation over the SMs. Its auto values were read back by
    bitwise-matching FA(auto) against FA(num_splits=s) on an A6000 (84 SMs, Hkv=2, decode) -- note
    several requested s collapse onto the same partition, so these are the canonical (lowest) members:

        zc = B*Hkv      2      4      8     16     32    64
        N=512           4      4      4      4      4     1
        N=2048         16     16     16      8      4     2
        N=4368         35     35     18      9      5     5

    Two rules reproduce that: (a) never finer than ~128 keys per split -- the small-zc column is
    exactly ceil(N/128); (b) otherwise ~1.5 waves of split-blocks over the SMs. This returns
    min(ceil(n_tiles/4), ceil(1.5*SM/zc)) with BLOCK_N=32, i.e. exactly those two rules:

        this fn      4/16/35   4/16/32   4/16/16   4/8/8   4/4/4   2/2/2

    Exactness is not the point and is not attainable from outside a compiled kernel: the error moves
    only 3.2% from 1 to 8 splits and ~1% from 8 to 16, so a few splits either way is well inside the
    level being matched. What matters is running the SAME structure (per-split fp32 m/l/acc + fp32 LSE
    combine) at the SAME order of split count, so the control is neither above nor below FA."""
    if n_sm is None:
        n_sm = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    coarse = max(1, -(-n_tiles // 4))                 # >= ~128 keys per split (FA's floor)
    waves = max(1, -(-3 * n_sm // (2 * max(zc, 1))))  # ~1.5 waves of split-blocks over the SMs
    return max(1, min(n_tiles, coarse, waves))


@triton.jit
def _flash_combine(
    Mp, Lp, Accp, Out, Z, Q_LEN,
    smz, sms, smm, slz, sls, slm, saz, sas, sam, sad, soz, son, sod,
    HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, N_SPLITS: tl.constexpr,
):
    """Log-sum-exp merge of the N_SPLITS partial (m,l,acc): global m=max_s m_s; out=(sum_s acc_s*
    exp(m_s-m)) / (sum_s l_s*exp(m_s-m)). Empty splits stored (-inf,0,0) contribute exp(-inf)=0."""
    pid_m = tl.program_id(0); pid_z = tl.program_id(1)
    offm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offd = tl.arange(0, BLOCK_D)
    mmask = offm < Q_LEN
    dmask = offd < HEAD_DIM
    m = tl.full([BLOCK_M], -float("inf"), tl.float32)
    for s in range(N_SPLITS):
        ms = tl.load(Mp + pid_z * smz + s * sms + offm * smm, mask=mmask, other=-float("inf"))
        m = tl.maximum(m, ms)
    l = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    for s in range(N_SPLITS):
        ms = tl.load(Mp + pid_z * smz + s * sms + offm * smm, mask=mmask, other=-float("inf"))
        ls = tl.load(Lp + pid_z * slz + s * sls + offm * slm, mask=mmask, other=0.0)
        accs = tl.load(Accp + pid_z * saz + s * sas + offm[:, None] * sam + offd[None, :] * sad,
                       mask=mmask[:, None] & dmask[None, :], other=0.0)
        alpha = tl.exp(ms - m)
        l += ls * alpha
        acc += accs * alpha[:, None]
    acc = acc / l[:, None]
    tl.store(Out + pid_z * soz + offm[:, None] * son + offd[None, :] * sod,
             acc.to(Out.dtype.element_ty), mask=mmask[:, None] & dmask[None, :])


_SPLIT_KERNEL_CACHE = {}


def _get_split_kernel(nmp_qk, w_qk, nmp_pv, w_pv, no_clamp, pack_dtype="bf16"):
    key = (nmp_qk, w_qk, nmp_pv, w_pv, no_clamp, pack_dtype)
    if key not in _SPLIT_KERNEL_CACHE:
        src = _gen_split_src(*key)
        fname = f"<flash_split_{key}>"
        linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
        ns = {"triton": triton, "tl": tl, "_bfp_scale": _bfp_scale}
        exec(compile(src, fname, "exec"), ns)
        _SPLIT_KERNEL_CACHE[key] = ns["_flash_split"]
    return _SPLIT_KERNEL_CACHE[key]


def flash_oz1fp_cg_splitkv(q, k, v, nmp, w, nmp_pv=None, w_pv=None, causal=True, sm_scale=None,
                           byte_split_style="all_signed_no_clamp", chunk_size=32, n_splits=8,
                           BLOCK_M=64, num_warps=4, num_stages=1, kv_lens=None, ozaki=True,
                           pv_split=False, pack_dtype="bf16"):
    """Split-KV / flash-decoding variant of flash_oz1fp_cg (non-cached, chunk=32). Partitions the kv
    loop into n_splits program-z slices to raise occupancy at decode (small B*Hkv grid), then merges
    the partials by log-sum-exp. Same numerics as flash_oz1fp_cg to fp-accumulation order (each split
    is whole BLOCK_N tiles -> identical per-tile block-FP). Intended for decode (q_len small); prefill
    already has enough parallelism. Falls back to n_splits=1 == the plain path (one slice).

    ozaki=False runs `_flash_exact_split_fwd`, the plain bf16 twin. For the exact path this is not an
    occupancy trick but a NUMERICAL match: vllm-flash-attn's decode is flash-decoding, and its split
    count is the entire reason it sat ~4.7% closer to fp64 than our single-pass exact kernel. Size
    n_splits with `fa_num_splits` to land on FA's level."""
    nmp_pv = nmp if nmp_pv is None else nmp_pv
    w_pv = w if w_pv is None else w_pv
    no_clamp = 1 if byte_split_style == "all_signed_no_clamp" else 0
    B, Hq, T, D = q.shape
    Hkv, N = k.shape[1], k.shape[2]
    G = Hq // Hkv
    assert Hq == Hkv * G, f"Hq={Hq} not a multiple of Hkv={Hkv}"
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    KQ = triton.next_power_of_2(min(chunk_size, D))
    assert KQ < D or not ozaki, "split-KV requires chunk_size < head_dim (the chunked path)"
    BLOCK_N = triton.next_power_of_2(chunk_size)
    Qrows = T * G
    qz = (q.reshape(B * Hkv, T, D) if G == 1 else
          q.reshape(B, Hkv, G, T, D).permute(0, 1, 3, 2, 4).reshape(B * Hkv, Qrows, D)).contiguous()
    kz, vz = (t.reshape(B * Hkv, N, D).contiguous() for t in (k, v))
    Zc = B * Hkv
    BLOCK_M = max(16, min(BLOCK_M, triton.next_power_of_2(Qrows)))
    BD = triton.next_power_of_2(D)
    Mp = torch.full((Zc, n_splits, Qrows), -float("inf"), device=q.device, dtype=torch.float32)
    Lp = torch.zeros((Zc, n_splits, Qrows), device=q.device, dtype=torch.float32)
    Accp = torch.zeros((Zc, n_splits, Qrows, D), device=q.device, dtype=torch.float32)
    o = torch.empty_like(qz)
    kern = (_get_split_kernel(nmp, w, nmp_pv, w_pv, no_clamp, pack_dtype) if ozaki
            else _flash_exact_split_fwd)
    has_kvlen = kv_lens is not None
    kvlen_z = (kv_lens.to(device=q.device, dtype=torch.int32).reshape(B).repeat_interleave(Hkv).contiguous()
               if has_kvlen else qz)
    extra = {} if ozaki else {"PV_SPLIT": bool(pv_split)}
    kern[(triton.cdiv(Qrows, BLOCK_M), Zc, n_splits)](
        qz, kz, vz, Mp, Lp, Accp, sm_scale, Zc, N, Qrows, N - T, kvlen_z,
        qz.stride(0), qz.stride(1), qz.stride(2), kz.stride(0), kz.stride(1), kz.stride(2),
        vz.stride(0), vz.stride(1), vz.stride(2),
        Mp.stride(0), Mp.stride(1), Mp.stride(2), Lp.stride(0), Lp.stride(1), Lp.stride(2),
        Accp.stride(0), Accp.stride(1), Accp.stride(2), Accp.stride(3),
        HEAD_DIM=D, BLOCK_D=BD, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        CAUSAL=causal, GQA_G=G, KQ=KQ, HAS_KVLEN=has_kvlen, N_SPLITS=n_splits,
        num_warps=num_warps, num_stages=num_stages, **extra,
    )
    _flash_combine[(triton.cdiv(Qrows, BLOCK_M), Zc)](
        Mp, Lp, Accp, o, Zc, Qrows,
        Mp.stride(0), Mp.stride(1), Mp.stride(2), Lp.stride(0), Lp.stride(1), Lp.stride(2),
        Accp.stride(0), Accp.stride(1), Accp.stride(2), Accp.stride(3),
        o.stride(0), o.stride(1), o.stride(2),
        HEAD_DIM=D, BLOCK_D=BD, BLOCK_M=BLOCK_M, N_SPLITS=n_splits, num_warps=4,
    )
    if G == 1:
        return o.reshape(B, Hq, T, D)
    return o.reshape(B, Hkv, T, G, D).permute(0, 1, 3, 2, 4).reshape(B, Hq, T, D)
