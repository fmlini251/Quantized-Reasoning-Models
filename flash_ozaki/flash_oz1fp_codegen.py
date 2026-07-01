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


def pack_plan(nmp, w):
    """Minimum signed g-bounded (g=8//w) rectangle cover of the kept (la,lb) digit-pair region.
    Returns [(i0,i1,j0,j1,sign)] -- each is one packed bf16 GEMM of A-digit-range [i0,i1] x
    B-digit-range [j0,j1] (span <= g -> super-digit < 2^8 -> bf16-exact), signed-combined. The place
    value 2^(w*(i0+j0)) is folded into the planes (absolute-place convention), so no external weight.
    Found by IDA* (branch on rectangles covering the first uncovered cell, both signs)."""
    key = (nmp, w)
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key]
    nD, drop, _ = oz1fp_params(nmp, w)
    g = max(1, 8 // w)
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
    assert all(chk[c] == (1 if c in kept else 0) for c in cells), f"bad plan nmp={nmp} w={w}"
    _PLAN_CACHE[key] = plan
    return plan


# --- kernel source generation -----------------------------------------------------------------
def _emit_peel(prefix, src, w, nD, no_clamp, ind):
    """Emit a single low->high digit peel producing NAMED place-folded bf16 planes
    {prefix}p0..{prefix}p{nD-1}, where {prefix}p{t} = signed_digit_t * 2^(w*t) (absolute place folded
    in -> bf16-exact). One peel, O(nD) -- vs the hand-rolled kernel's O(nD^2) repeated _signed_digit."""
    base, half = (1 << w), (1 << (w - 1))
    L, cur = [], src
    for t in range(nD):
        if t == nD - 1 and no_clamp:
            d = cur                                              # top digit = unclamped remainder
        else:
            L.append(f"{ind}{prefix}_lo = {cur} & {base - 1}")
            L.append(f"{ind}{prefix}d{t} = tl.where({prefix}_lo >= {half}, {prefix}_lo - {base}, {prefix}_lo)")
            d = f"{prefix}d{t}"
            if t != nD - 1:
                L.append(f"{ind}{prefix}_c{t} = ({cur} - {prefix}d{t}) >> {w}")
                cur = f"{prefix}_c{t}"
        L.append(f"{ind}{prefix}p{t} = ({d} * {1 << (w * t)}).to(tl.bfloat16)")
    return "\n".join(L)


def _super(prefix, i0, i1):
    return "(" + " + ".join(f"{prefix}p{t}" for t in range(i0, i1 + 1)) + ")"


def _emit_dots(acc, plan, a, b, trans_b, ind):
    """Emit one explicit signed tl.dot per plan rectangle (the fused-kernel analog of the GEMM's
    fixed named slots). place 2^(w*(i0+j0)) is already folded into the planes -> no external weight."""
    L = []
    for (i0, i1, j0, j1, sgn) in plan:
        bexpr = f"tl.trans({_super(b, j0, j1)})" if trans_b else _super(b, j0, j1)
        op = "+=" if sgn > 0 else "-="
        L.append(f"{ind}{acc} {op} tl.dot({_super(a, i0, i1)}, {bexpr}, out_dtype=tl.float32)")
    return "\n".join(L)


def _gen_src(nmp_qk, w_qk, nmp_pv, w_pv, no_clamp, chunked):
    nD_qk = oz1fp_params(nmp_qk, w_qk)[0]
    nD_pv = oz1fp_params(nmp_pv, w_pv)[0]
    ib_qk = w_qk * nD_qk - 1                                  # production int_bits = w*nD-1
    ib_pv = w_pv * nD_pv - 1
    plan_qk = pack_plan(nmp_qk, w_qk)
    plan_pv = pack_plan(nmp_pv, w_pv)
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
{_emit_peel("q", "qI", w_qk, nD_qk, no_clamp, I)}'''
        qk_body = f'''        k = tl.load(K + pid_z * skz + offn[:, None] * skn + offd[None, :] * skd,
                    mask=nmask[:, None] & dmask[None, :], other=0.0)
        ksc = _bfp_scale(tl.max(tl.abs(k), axis=1).to(tl.float32), {ib_qk})
        kI = (k / ksc[:, None] + tl.where(k >= 0, 0.5, -0.5)).to(tl.int32)
        kI = tl.minimum(tl.maximum(kI, {lo_qk}), {hi_qk})
{_emit_peel("k", "kI", w_qk, nD_qk, no_clamp, L)}
        cacc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
{_emit_dots("cacc", plan_qk, "q", "k", True, L)}
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
{_emit_peel("q", "qI", w_qk, nD_qk, no_clamp, L2)}
            kc = tl.load(K + pid_z * skz + offn[:, None] * skn + offc[None, :] * skd,
                         mask=nmask[:, None] & cmask[None, :], other=0.0)
            ksc = _bfp_scale(tl.max(tl.abs(kc), axis=1).to(tl.float32), {ib_qk})
            kI = (kc / ksc[:, None] + tl.where(kc >= 0, 0.5, -0.5)).to(tl.int32)
            kI = tl.minimum(tl.maximum(kI, {lo_qk}), {hi_qk})
{_emit_peel("k", "kI", w_qk, nD_qk, no_clamp, L2)}
            cacc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
{_emit_dots("cacc", plan_qk, "q", "k", True, L2)}
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
        psc = _bfp_scale(tl.max(p, axis=1), {ib_pv})
        vsc = _bfp_scale(tl.max(tl.abs(v), axis=0).to(tl.float32), {ib_pv})
        pI = (p / psc[:, None] + 0.5).to(tl.int32)
        pI = tl.minimum(pI, {hi_pv})
        vI = (v / vsc[None, :] + tl.where(v >= 0, 0.5, -0.5)).to(tl.int32)
        vI = tl.minimum(tl.maximum(vI, {lo_pv}), {hi_pv})
{_emit_peel("pp", "pI", w_pv, nD_pv, no_clamp, L)}
{_emit_peel("vv", "vI", w_pv, nD_pv, no_clamp, L)}
        pv = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
{_emit_dots("pv", plan_pv, "pp", "vv", False, L)}
        pv = pv * psc[:, None] * vsc[None, :]
        acc = acc * alpha[:, None] + pv
        m_i = m_new'''

    src = f'''
@triton.jit
def _flash_cg(
    Q, K, V, Out, sm_scale, Z, N_CTX, Q_LEN, Q_OFF,
    sqz, sqn, sqd, skz, skn, skd, svz, svn, svd, soz, son, sod,
    HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, GQA_G: tl.constexpr, KQ: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_z = tl.program_id(1)
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
        n_end = tl.minimum(Q_OFF + last_m // GQA_G + 1, N_CTX)
    else:
        n_end = N_CTX
    for n0 in range(0, n_end, BLOCK_N):
        offn = n0 + tl.arange(0, BLOCK_N)
        nmask = offn < N_CTX
{qk_body}
{tail}
    acc = acc / l_i[:, None]
    tl.store(Out + pid_z * soz + offm[:, None] * son + offd[None, :] * sod,
             acc.to(Out.dtype.element_ty), mask=qmask)
'''
    return src


_KERNEL_CACHE = {}


def _get_kernel(nmp_qk, w_qk, nmp_pv, w_pv, no_clamp, chunked):
    key = (nmp_qk, w_qk, nmp_pv, w_pv, no_clamp, chunked)
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
    Q, K, V, Out, sm_scale, Z, N_CTX, Q_LEN, Q_OFF,
    sqz, sqn, sqd, skz, skn, skd, svz, svn, svd, soz, son, sod,
    HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, GQA_G: tl.constexpr, KQ: tl.constexpr,
):
    """Plain bf16 flash attention (no ozaki) -- the flash-exact baseline the benches compare against.
    Same online-softmax / causal / GQA-fold structure as the generated ozaki kernel, so the two are
    apples-to-apples; only the QK/PV dots differ (single bf16 dot vs digit-plane plan)."""
    pid_m = tl.program_id(0); pid_z = tl.program_id(1)
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
        n_end = tl.minimum(Q_OFF + last_m // GQA_G + 1, N_CTX)
    else:
        n_end = N_CTX
    for n0 in range(0, n_end, BLOCK_N):
        offn = n0 + tl.arange(0, BLOCK_N)
        nmask = offn < N_CTX
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
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(Out + pid_z * soz + offm[:, None] * son + offd[None, :] * sod,
             acc.to(Out.dtype.element_ty), mask=qmask)


def flash_oz1fp_cg(q, k, v, nmp, w, nmp_pv=None, w_pv=None, causal=True, sm_scale=None, ozaki=True,
                   byte_split_style="all_signed_no_clamp", chunk_size=None, BLOCK_M=64, BLOCK_N=64,
                   num_warps=4, num_stages=1):
    """Code-generated Flash-Ozaki1_fp. q:[B,Hq,T,D]; k,v:[B,Hkv,N,D] bf16 (MHA Hq==Hkv; GQA Hq=Hkv*G
    folds the G group-heads into the query-row dim). nmp/w drive QK^T; nmp_pv/w_pv (default = nmp/w)
    drive P@V INDEPENDENTLY -- QK-nmp != PV-nmp is supported. Optimal pack plan + single-peel planes
    are baked into the kernel source per (nmp_qk,w_qk,nmp_pv,w_pv,style). ozaki=False runs the plain
    bf16 flash path (same softmax/GQA machinery) -- the flash-exact baseline.

    chunk_size: the block-FP chunk applied to EVERY GEMM reduction, matching production. QK's head_dim
    reduction is split into KQ=next_pow2(min(chunk_size,D))-wide chunks, and PV's kv reduction chunk is
    the kv tile, so BLOCK_N is set to next_pow2(chunk_size). chunk_size=None keeps the fast un-chunked
    path (QK: one block-FP scale over the full head_dim; PV: chunk = BLOCK_N). Best power-of-2."""
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
    kern = _get_kernel(nmp, w, nmp_pv, w_pv, no_clamp, chunked) if ozaki else _flash_exact_fwd

    def _launch(bm):
        kern[(triton.cdiv(Qrows, bm), B * Hkv)](
            qz, kz, vz, o, sm_scale, B * Hkv, N, Qrows, N - T,
            qz.stride(0), qz.stride(1), qz.stride(2), kz.stride(0), kz.stride(1), kz.stride(2),
            vz.stride(0), vz.stride(1), vz.stride(2), o.stride(0), o.stride(1), o.stride(2),
            HEAD_DIM=D, BLOCK_D=triton.next_power_of_2(D), BLOCK_M=bm, BLOCK_N=BLOCK_N,
            CAUSAL=causal, GQA_G=G, KQ=KQ, num_warps=num_warps, num_stages=num_stages,
        )
    _run_with_oom_retry(_launch, BLOCK_M)
    if G == 1:
        return o.reshape(B, Hq, T, D)
    return o.reshape(B, Hkv, T, G, D).permute(0, 1, 3, 2, 4).reshape(B, Hq, T, D)


# --- pre-encoded KV cache (the attention analog of weight_cache) -------------------------------
def _gen_cached_src(nmp_qk, w_qk, nmp_pv, w_pv, no_clamp, chunked):
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
    plan_qk = pack_plan(nmp_qk, w_qk)
    plan_pv = pack_plan(nmp_pv, w_pv)
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
{_emit_peel("q", "qI", w_qk, nD_qk, no_clamp, I)}'''
        qk_body = f'''        ksc = tl.load(Ks + pid_z * sksz + offn * sksn, mask=nmask, other=0.0)
{kload}
        cacc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
{_emit_dots("cacc", plan_qk, "q", "k", True, L)}
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
{_emit_peel("q", "qI", w_qk, nD_qk, no_clamp, L2)}
            ksc = tl.load(Ks + pid_z * sksz + (dc // KQ) * skscd + offn * sksn, mask=nmask, other=0.0)
{kload}
            cacc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
{_emit_dots("cacc", plan_qk, "q", "k", True, L2)}
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
        psc = _bfp_scale(tl.max(p, axis=1), {ib_pv})
        pI = (p / psc[:, None] + 0.5).to(tl.int32)
        pI = tl.minimum(pI, {hi_pv})
{_emit_peel("pp", "pI", w_pv, nD_pv, no_clamp, L)}
{vload}
        pv = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
{_emit_dots("pv", plan_pv, "pp", "vv", False, L)}
        pv = pv * psc[:, None] * vsc[None, :]
        acc = acc * alpha[:, None] + pv
        m_i = m_new'''

    src = f'''
@triton.jit
def _flash_cg_cached(
    Q, Kp, Ks, Vp, Vs, Out, sm_scale, Z, N_CTX, Q_LEN, Q_OFF,
    sqz, sqn, sqd, skz, skt, skn, skd, sksz, skscd, sksn,
    svz, svt, svn, svd, svsz, svsc, svsd, soz, son, sod,
    HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, GQA_G: tl.constexpr, KQ: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_z = tl.program_id(1)
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
        n_end = tl.minimum(Q_OFF + last_m // GQA_G + 1, N_CTX)
    else:
        n_end = N_CTX
    for n0 in range(0, n_end, BLOCK_N):
        offn = n0 + tl.arange(0, BLOCK_N)
        nmask = offn < N_CTX
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
                          num_warps=4, num_stages=1):
    """Cached-KV codegen flash. q:[B,Hq,T,D]; kv = encode_kv(..., chunk_size=chunk_size) (Hkv kv heads,
    length N). MHA: Hq==Hkv. GQA: Hq=Hkv*G -> the G query heads sharing each cached kv head fold into
    the query-row dim, so one cache tile feeds all G. Prefill T==N; decode/chunked T<N (q token i at abs
    kv pos N-T+i). EXACT vs non-cached flash_oz1fp_cg for the same config. chunk_size MUST match the
    encode_kv call: it chunks K's head_dim (KQ, via k_scale[Z,nchd,N]) and V's kv (BLOCK_N)."""
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

    def _launch(bm):
        kern[(triton.cdiv(Qrows, bm), Zc)](
            qz, k_pl, k_scale, v_pl, v_scale, o, sm_scale, Zc, N, Qrows, N - T,
            qz.stride(0), qz.stride(1), qz.stride(2),
            k_pl.stride(0), k_pl.stride(1), k_pl.stride(2), k_pl.stride(3),
            k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
            v_pl.stride(0), v_pl.stride(1), v_pl.stride(2), v_pl.stride(3),
            v_scale.stride(0), v_scale.stride(1), v_scale.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            HEAD_DIM=D, BLOCK_D=triton.next_power_of_2(D), BLOCK_M=bm, BLOCK_N=BLOCK_N,
            CAUSAL=causal, GQA_G=G, KQ=KQ, num_warps=num_warps, num_stages=num_stages,
        )
    _run_with_oom_retry(_launch, BLOCK_M)
    if G == 1:
        return o.reshape(B, Hq, T, D)
    return o.reshape(B, Hkv, T, G, D).permute(0, 1, 3, 2, 4).reshape(B, Hq, T, D)
