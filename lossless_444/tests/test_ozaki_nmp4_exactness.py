"""C2-lite: fmt_lib-stored 4-bit operands through the ozaki1 GEMM at nmp=4 (FIXED), w=4.

nmp=4 -> oz1fp_params: nD=2 (full digit grid), int_bits=7.  Digit-collapse claim (H0/theory SS1):
any 4-bit lattice format re-encodes EXACTLY at nD=2 -- including the mxint4 "-8 at block max"
1-LSB boundary found at nD=1 (fmt_lib G2), because the frexp int_bits=7 prealign leaves 4 bits of
headroom.  So kernel output vs fp64 reference of the dequantized operands must sit at fp32
combine-rounding level (~1e-6 rel), while a bf16 non-lattice control shows the ordinary nmp=4
emulation error, orders of magnitude larger.

GPU required.  Run: python lossless_444/tests/test_ozaki_nmp4_exactness.py
"""
import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lossless_444.scripts import fmt_lib as F
from flash_ozaki.verification.standalone_oz_gemm import oz1fp_gemm_cg
from flash_ozaki.oz1fp_triton import oz1fp_params

NMP, W, CHUNK = 4, 4, 32
Z, M, K, N = 1, 64, 256, 64
DEV = "cuda"

torch.manual_seed(0)


def _stored(x, fmt, **kw):
    """Quantize to a storage format and return the dequantized (grid-resident) tensor."""
    q = F.quantize(x, F.QuantConfig(fmt, **kw))
    return F.dequantize(q)


def _gemm_relerr(a, b):
    """ozaki nmp=4 GEMM vs fp64 reference of the SAME (dequantized) operands."""
    A = a.reshape(Z, M, K).float().to(DEV)
    B = b.reshape(Z, K, N).float().to(DEV)
    C = oz1fp_gemm_cg(A, B, NMP, W, chunk=CHUNK)
    ref = (A.double() @ B.double()).float()
    return ((C - ref).abs().max() / ref.abs().max()).item()


def main():
    nD, drop, ib = oz1fp_params(NMP, W)
    assert (nD, drop, ib) == (2, 0, 7), f"unexpected nmp=4 params {(nD, drop, ib)}"
    print(f"nmp={NMP} w={W} -> nD={nD} (full), int_bits={ib}")

    results = {}

    # (a) A,B both mxint4-stored (the C3 A4/KV4 shape)
    a = _stored(torch.randn(M, K) * 3, "mxint4")
    b = _stored(torch.randn(K, N) * 3, "mxint4")
    results["mxint4 @ mxint4"] = _gemm_relerr(a, b)

    # (b) mixed-dtype operands: A mxint4 x B mxfp4 (the mixed-ALU selling point)
    b2 = _stored(torch.randn(K, N) * 3, "mxfp4")
    results["mxint4 @ mxfp4"] = _gemm_relerr(a, b2)

    # (c) adversarial -8-at-block-max: every block's max-magnitude element is code -8
    #     (the nD=1 boundary case; must be exact at nD=2)
    codes = torch.randint(-7, 8, (M, K)).float()
    codes[:, ::CHUNK] = -8.0                                  # -8 is every A-block's max
    scales = torch.exp2(torch.randint(-3, 4, (M, K // CHUNK)).float())
    aneg = (codes.reshape(M, -1, CHUNK) * scales.unsqueeze(-1)).reshape(M, K)
    codes_b = torch.randint(-7, 8, (K, N)).float()
    codes_b[::CHUNK, :] = -8.0                                # -8 is every B-chunk's max
    results["-8-boundary mxint4 @ int"] = _gemm_relerr(aneg, codes_b)

    # (d) uint4-stored A (unsigned lattice rides the signed datapath: 0..15 ints are plain ints)
    au = _stored(torch.rand(M, K) * 5, "uint4")
    results["uint4 @ mxint4"] = _gemm_relerr(au, b)

    # (e) control: bf16 non-lattice operands -> ordinary nmp=4 emulation error (NOT exact)
    ac = (torch.randn(M, K) * 3).to(torch.bfloat16).float()
    bc = (torch.randn(K, N) * 3).to(torch.bfloat16).float()
    results["control bf16 (non-stored)"] = _gemm_relerr(ac, bc)

    print(f"\n{'case':32s} rel_err")
    for k, v in results.items():
        print(f"{k:32s} {v:.3e}")

    exact_bound = 5e-6                       # fp32 combine rounding across K/CHUNK partial sums
    fails = []
    for k, v in results.items():
        if k.startswith("control"):
            if v < 10 * exact_bound:
                fails.append(f"{k}: control unexpectedly exact ({v:.1e}) -- test not sensitive")
        elif v > exact_bound:
            fails.append(f"{k}: rel_err {v:.3e} > {exact_bound:.0e} (digit collapse broken)")
    if fails:
        print("\nFAIL"); [print(" ", f) for f in fails]; sys.exit(1)
    print(f"\nPASS: all stored-lattice GEMMs exact at nmp={NMP} (<= {exact_bound:.0e}); "
          f"control shows normal emulation error")


if __name__ == "__main__":
    main()
