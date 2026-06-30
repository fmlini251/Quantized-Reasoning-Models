"""Flash-Ozaki1_fp with a PRE-ENCODED KV cache (kv_cache=True) -- the attention analog of linear
weight_cache. K and V are static once written, so we block-FP-encode them into ozaki1_fp digit
planes (+ scales) ONCE and the flash kernel LOADS those, skipping the per-tile K/V encode (the
amax-reduce + round + signed-digit-split + cast that dominates the nmp=1 overhead). Q and P are
recomputed per step (P = current softmax weights, not cacheable), so they are still encoded inline.

K is encoded per-row over head_dim (chunk=D, matches the non-cached kernel). V is encoded per-dim
over the full sequence (chunk=N) for a clean cache (vs the non-cached kernel's per-tile V scale --
a small block-FP granularity difference; validated vs exact below). Digit planes carry the place
value 2^(w*t) folded in (exact in bf16). Memory ~nD x the KV (the weight_cache-style tradeoff).
"""
import torch
import triton
import triton.language as tl

from flash_ozaki.oz1fp_triton import _signed_digit, oz1fp_params


# ---------------- one-time KV encode (the "cache build") ----------------
def _digit_planes(xI, w, nD):
    """int64 xI -> [nD, ...] bf16 planes with place folded (digit_t * 2^(w*t)); no_clamp split."""
    base, half = (1 << w), (1 << (w - 1))
    planes, cur = [], xI.clone()
    for t in range(nD):
        if t == nD - 1:
            d = cur                                            # top digit: unclamped remainder
        else:
            low = cur & (base - 1)
            d = torch.where(low >= half, low - base, low)
            cur = (cur - d) >> w
        planes.append((d.to(torch.float32) * (2.0 ** (w * t))).to(torch.bfloat16))
    return torch.stack(planes, dim=1)                          # [Z, nD, ...]


def encode_kv(k, v, nmp, w):
    """k,v: [B,H,N,D] bf16 -> (k_dig[Z,nD,N,D], k_scale[Z,N], v_dig[Z,nD,N,D], v_scale[Z,D])."""
    B, H, N, D = k.shape
    Z = B * H
    nD, _, int_bits = oz1fp_params(nmp, w)
    maxmag = float((1 << int_bits) - 1)
    kf = k.reshape(Z, N, D).float(); vf = v.reshape(Z, N, D).float()
    k_scale = kf.abs().amax(-1, keepdim=True) / maxmag + 1e-20           # [Z,N,1] per-row over D
    k_dig = _digit_planes(torch.round(kf / k_scale).to(torch.int64), w, nD)  # [Z,nD,N,D]
    v_scale = vf.abs().amax(1, keepdim=True) / maxmag + 1e-20            # [Z,1,D] per-dim over N
    v_dig = _digit_planes(torch.round(vf / v_scale).to(torch.int64), w, nD)
    return (k_dig.contiguous(), k_scale.squeeze(-1).contiguous(),
            v_dig.contiguous(), v_scale.squeeze(1).contiguous())


@triton.jit
def _flash_cached_fwd(
    Q, Kd, Ks, Vd, Vs, Out, sm_scale, Z, N_CTX,
    sqz, sqn, sqd, skz, skt, skn, skd, sksz, sksn,
    svz, svt, svn, svd, svsz, svsd, soz, son, sod,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, W: tl.constexpr, ND: tl.constexpr, DROP: tl.constexpr, MAXMAG: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_z = tl.program_id(1)
    offm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offd = tl.arange(0, HEAD_DIM)
    qmask = offm[:, None] < N_CTX
    q = tl.load(Q + pid_z * sqz + offm[:, None] * sqn + offd[None, :] * sqd, mask=qmask, other=0.0)
    qsc = tl.max(tl.abs(q), axis=1).to(tl.float32) / MAXMAG + 1e-20      # Q encoded inline (cheap)
    qI = (q / qsc[:, None] + tl.where(q >= 0, 0.5, -0.5)).to(tl.int32)
    vsc = tl.load(Vs + pid_z * svsz + offd * svsd)                       # [D] global per-dim, load once

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    n_end = (pid_m + 1) * BLOCK_M if CAUSAL else N_CTX
    for n0 in range(0, n_end, BLOCK_N):
        offn = n0 + tl.arange(0, BLOCK_N)
        nmask = offn < N_CTX
        ksc = tl.load(Ks + pid_z * sksz + offn * sksn, mask=nmask, other=0.0)   # cached K scale
        qk = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for la in range(ND):
            qd = (_signed_digit(qI, la, W, ND) * (1 << (W * la))).to(tl.bfloat16)
            for lb in range(ND):
                if W * (la + lb) >= DROP:                                    # load cached K digit lb (place folded)
                    kd = tl.load(Kd + pid_z * skz + lb * skt + offn[:, None] * skn + offd[None, :] * skd,
                                 mask=nmask[:, None], other=0.0)
                    qk += tl.dot(qd, tl.trans(kd), out_dtype=tl.float32)
        qk = qk * qsc[:, None] * ksc[None, :] * sm_scale
        qk = tl.where(nmask[None, :], qk, -float("inf"))
        if CAUSAL:
            qk = tl.where(offm[:, None] >= offn[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        psc = tl.max(p, axis=1) / MAXMAG + 1e-20                            # P encoded inline (not cacheable)
        pI = (p / psc[:, None] + 0.5).to(tl.int32)
        pv = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
        for la in range(ND):
            pd = (_signed_digit(pI, la, W, ND) * (1 << (W * la))).to(tl.bfloat16)
            for lb in range(ND):
                if W * (la + lb) >= DROP:                                    # load cached V digit lb
                    vd = tl.load(Vd + pid_z * svz + lb * svt + offn[:, None] * svn + offd[None, :] * svd,
                                 mask=nmask[:, None], other=0.0)
                    pv += tl.dot(pd, vd, out_dtype=tl.float32)
        pv = pv * psc[:, None] * vsc[None, :]
        acc = acc * alpha[:, None] + pv
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(Out + pid_z * soz + offm[:, None] * son + offd[None, :] * sod,
             acc.to(Out.dtype.element_ty), mask=qmask)


def flash_oz1fp_cached(q, kv, nmp, w, causal=True, sm_scale=None, BLOCK_M=64, BLOCK_N=64):
    """q: [B,H,N,D]; kv = encode_kv(...) tuple. Uses the pre-encoded K/V (no in-kernel K/V encode)."""
    B, H, N, D = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    nD, drop, int_bits = oz1fp_params(nmp, w)
    k_dig, k_scale, v_dig, v_scale = kv
    qz = q.reshape(B * H, N, D).contiguous()
    o = torch.empty_like(qz)
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    _flash_cached_fwd[grid](
        qz, k_dig, k_scale, v_dig, v_scale, o, sm_scale, B * H, N,
        qz.stride(0), qz.stride(1), qz.stride(2),
        k_dig.stride(0), k_dig.stride(1), k_dig.stride(2), k_dig.stride(3), k_scale.stride(0), k_scale.stride(1),
        v_dig.stride(0), v_dig.stride(1), v_dig.stride(2), v_dig.stride(3), v_scale.stride(0), v_scale.stride(1),
        o.stride(0), o.stride(1), o.stride(2),
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, CAUSAL=causal,
        W=w, ND=nD, DROP=drop, MAXMAG=float((1 << int_bits) - 1), num_warps=4, num_stages=1,
    )
    return o.reshape(B, H, N, D)
