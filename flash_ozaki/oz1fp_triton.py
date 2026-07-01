"""Shared ozaki1_fp parameter math.

`oz1fp_params` maps an (nmp, w) config to (nD, drop, int_bits) and is the one piece the whole
flash_ozaki suite needs in common. The bf16 byte-plane GEMM/attention kernels themselves live in
flash_oz1fp_codegen.py (fused flash + cached KV) and verification/standalone_oz_gemm.py (isolated
GEMM), both on the production-faithful block-FP scale (frexp int_bits, clamp). The earlier
hand-rolled kernels here (_po2/_maxmag scale, inline O(nD^2) digit peel + copy-pasted rectangle
cover) were superseded by that codegen path and removed.
"""
import math


def oz1fp_params(nmp, w):
    """(nmp, w) -> (nD, drop, int_bits). full nmp=nD^2 (drop=0, keep every digit-pair); triangular
    nmp=nD(nD+1)/2 (drop=w*(nD-1), keep w*(la+lb) >= drop). int_bits = w*nD - 1."""
    r = math.isqrt(nmp)
    if r * r == nmp:
        nD, drop = r, 0
    else:
        nD = (math.isqrt(8 * nmp + 1) - 1) // 2
        assert nD * (nD + 1) // 2 == nmp, f"nmp={nmp} is neither full (nD^2) nor triangular (T_nD)"
        drop = w * (nD - 1)
    return nD, drop, w * nD - 1
