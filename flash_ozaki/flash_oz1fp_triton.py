"""Flash-Ozaki1_fp: FlashAttention with the **bf16 byte-plane ozaki1_fp GEMM** fused into the tile
dots (QK^T and P@V). Online softmax + fp32 accumulation are untouched (flash numerics preserved);
only the two matmuls go through the ozaki1_fp digit-pair decomposition (validated faithful to the
production `batched_gemm` in oz1fp_triton.py). Parameterized by (w, nmp) -> matches the sweep configs.

Per tile: each operand is block-FP scaled (per-row over the dot reduction), split into nD signed
w-bit digits (bf16), and the kept digit-pairs are bf16-dotted with the place value 2^(W*(la+lb))
folded into the digits (exact), fp32-accumulated. Block-FP chunk = full reduction (head_dim for
QK^T, kv-tile for P@V); production uses k=32 (fidelity refinement, same mechanism).
"""
import torch
import triton
import triton.language as tl

from flash_ozaki.oz1fp_triton import _signed_digit, oz1fp_params


@triton.jit
def _flash_oz1fp_fwd(
    Q, K, V, Out, sm_scale, Z, N_CTX,
    stride_qz, stride_qn, stride_qd, stride_kz, stride_kn, stride_kd,
    stride_vz, stride_vn, stride_vd, stride_oz, stride_on, stride_od,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, OZAKI: tl.constexpr,
    W: tl.constexpr, ND: tl.constexpr, DROP: tl.constexpr, MAXMAG: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_z = tl.program_id(1)
    offm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offd = tl.arange(0, HEAD_DIM)
    qmask = offm[:, None] < N_CTX
    q = tl.load(Q + pid_z * stride_qz + offm[:, None] * stride_qn + offd[None, :] * stride_qd,
                mask=qmask, other=0.0)
    if OZAKI:
        qsc = tl.max(tl.abs(q), axis=1).to(tl.float32) / MAXMAG + 1e-20
        qI = (q / qsc[:, None] + tl.where(q >= 0, 0.5, -0.5)).to(tl.int32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    n_end = (pid_m + 1) * BLOCK_M if CAUSAL else N_CTX
    for n0 in range(0, n_end, BLOCK_N):
        offn = n0 + tl.arange(0, BLOCK_N)
        nmask = offn < N_CTX
        k = tl.load(K + pid_z * stride_kz + offn[:, None] * stride_kn + offd[None, :] * stride_kd,
                    mask=nmask[:, None], other=0.0)
        if OZAKI:
            ksc = tl.max(tl.abs(k), axis=1).to(tl.float32) / MAXMAG + 1e-20
            kI = (k / ksc[:, None] + tl.where(k >= 0, 0.5, -0.5)).to(tl.int32)
            qk = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
            for la in range(ND):
                qd = (_signed_digit(qI, la, W, ND) * (1 << (W * la))).to(tl.bfloat16)
                for lb in range(ND):
                    if W * (la + lb) >= DROP:
                        kd = (_signed_digit(kI, lb, W, ND) * (1 << (W * lb))).to(tl.bfloat16)
                        qk += tl.dot(qd, tl.trans(kd), out_dtype=tl.float32)
            qk = qk * qsc[:, None] * ksc[None, :]
        else:
            qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale
        qk = tl.where(nmask[None, :], qk, -float("inf"))
        if CAUSAL:
            qk = tl.where(offm[:, None] >= offn[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(V + pid_z * stride_vz + offn[:, None] * stride_vn + offd[None, :] * stride_vd,
                    mask=nmask[:, None], other=0.0)
        if OZAKI:
            psc = tl.max(p, axis=1) / MAXMAG + 1e-20                 # p>=0, fp32
            pI = (p / psc[:, None] + 0.5).to(tl.int32)
            vsc = tl.max(tl.abs(v), axis=0).to(tl.float32) / MAXMAG + 1e-20
            vI = (v / vsc[None, :] + tl.where(v >= 0, 0.5, -0.5)).to(tl.int32)
            pv = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
            for la in range(ND):
                pd = (_signed_digit(pI, la, W, ND) * (1 << (W * la))).to(tl.bfloat16)
                for lb in range(ND):
                    if W * (la + lb) >= DROP:
                        vd = (_signed_digit(vI, lb, W, ND) * (1 << (W * lb))).to(tl.bfloat16)
                        pv += tl.dot(pd, vd, out_dtype=tl.float32)
            pv = pv * psc[:, None] * vsc[None, :]
        else:
            pv = tl.dot(p.to(v.dtype), v)

        acc = acc * alpha[:, None] + pv
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(Out + pid_z * stride_oz + offm[:, None] * stride_on + offd[None, :] * stride_od,
             acc.to(Out.dtype.element_ty), mask=qmask)


def flash_oz1fp(q, k, v, nmp, w, causal=True, ozaki=True, sm_scale=None,
                BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=1):
    """q,k,v: [B,H,N,D] bf16 (MHA). ozaki=True -> bf16 byte-plane ozaki1_fp tile dots; False -> exact."""
    B, H, N, D = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    nD, drop, int_bits = oz1fp_params(nmp, w)
    maxmag = float((1 << int_bits) - 1)
    qz, kz, vz = (t.reshape(B * H, N, D).contiguous() for t in (q, k, v))
    o = torch.empty_like(qz)
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    _flash_oz1fp_fwd[grid](
        qz, kz, vz, o, sm_scale, B * H, N,
        qz.stride(0), qz.stride(1), qz.stride(2), kz.stride(0), kz.stride(1), kz.stride(2),
        vz.stride(0), vz.stride(1), vz.stride(2), o.stride(0), o.stride(1), o.stride(2),
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, CAUSAL=causal, OZAKI=ozaki,
        W=w, ND=nD, DROP=drop, MAXMAG=maxmag, num_warps=num_warps, num_stages=num_stages,
    )
    return o.reshape(B, H, N, D)
