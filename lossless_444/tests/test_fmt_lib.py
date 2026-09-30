"""G2 gate: fmt_lib round-trip / exactness / STE / FlatQuant-plugin unit tests (mixed_4bit.md SS0.4).

Runs standalone (`python lossless_444/tests/test_fmt_lib.py`) or under pytest. CPU-only.
"""
import math
import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lossless_444.scripts import fmt_lib as F

torch.manual_seed(0)


def _rand(rows=64, k=96, scale=3.0):
    return torch.randn(rows, k) * scale


# ---------------------------------------------------------------- round-trip bounds

def test_roundtrip_all_formats():
    x = _rand()
    for fid in ["int4_pc", "int4_pt", "int4_g128", "nvint4", "mxint4", "mxfp4",
                "nvfp4", "e1m2", "mixfp4", "nf4"]:
        y = F.fake_quant_ste(x, F.QuantConfig(fid))
        assert torch.isfinite(y).all(), fid
        rel = (y - x).pow(2).mean() / x.pow(2).mean()
        assert rel < 0.05, f"{fid}: rel MSE {rel:.4f} out of bound"
        # idempotence: re-quantizing the dequantized tensor is exact -- except pow2-scale
        # signed formats when code -8 is the block max (frexp scale bumps one octave; same
        # 1-LSB boundary as the production ozaki re-encode), so allow <=1 LSB drift there.
        y2 = F.fake_quant_ste(y.detach(), F.QuantConfig(fid))
        if fid == "mxint4":
            q = F.quantize(x, F.QuantConfig(fid))
            assert (y2 - y).abs().max() <= q.scales.max() + 1e-6, f"{fid}: >1 LSB requant drift"
        else:
            assert torch.allclose(y2, y, atol=0, rtol=0), f"{fid}: not idempotent"


def test_scales_finite_and_positive():
    x = torch.zeros(4, 64)  # all-zero blocks -> canonical scale 1, exact zero
    for fid in ["int4_g128", "mxint4", "mxfp4", "nvfp4", "mixfp4"]:
        q = F.quantize(x, F.QuantConfig(fid))
        assert torch.isfinite(q.scales).all() and (q.scales > 0).all(), fid
        assert F.dequantize(q).abs().max() == 0, fid


# ---------------------------------------------------------------- lattice exactness

def test_grid_exactness_pow2_formats():
    """Values already on the (pow2-scaled) lattice survive quant-dequant bit-exactly --
    the storage premise of digit collapse (theory SS1)."""
    # NOTE mxint4 ints drawn from [-7,7]: a -8 code at the block max bumps the frexp scale one
    # octave (amax=2^ib boundary) -- the digit-collapse-safe stored range. C2 must re-verify
    # this boundary end-to-end on the kernel side.
    for fid, gen in [
        ("mxint4", lambda: torch.randint(-7, 8, (16, 64)).float()),
        ("uint4",  lambda: torch.randint(0, 16, (16, 64)).float()),
        ("mxfp4",  lambda: torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])[
            torch.randint(0, 8, (16, 64))] * torch.randint(1, 3, (16, 64)).float().sign()),
    ]:
        base = gen()
        for e in (-3, 0, 5):
            x = base * (2.0 ** e)
            y = F.fake_quant_ste(x, F.QuantConfig(fid))
            assert torch.equal(y, x), f"{fid} not exact at 2^{e}"


def test_mxint4_equals_ozaki_encode():
    """MXINT4 == ozaki 1-digit encode: same pow2 scale (frexp int_bits=3) and same codes
    under the production rounding (half-away)."""
    sys.path.insert(0, _REPO)
    from flash_ozaki.flash_oz1fp_codegen import _bfp_scale_torch
    x = _rand(32, 64)
    q = F.quantize(x, F.QuantConfig("mxint4", rounding="half_away"))
    xb = x.reshape(32, 2, 32)
    s_oz = _bfp_scale_torch(xb.abs().amax(-1), 3)
    assert torch.equal(q.scales, s_oz), "scale rule differs from production _bfp_scale"
    rnd = lambda t: torch.trunc(t + torch.where(t >= 0, 0.5, -0.5))
    codes_oz = rnd(xb / s_oz.unsqueeze(-1)).clamp(-8, 7)
    assert torch.equal(q.codes.float(), codes_oz), "codes differ from ozaki encode"


# ---------------------------------------------------------------- digit planes

def test_e2m1_digit_planes():
    """E2M1 = 2 planes; 2nd plane ternary, nonzero exactly on aligned ints +-8/+-12
    (codebook values +-4/+-6, theory SS1)."""
    lv = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    vals = torch.cat([lv, -lv]).reshape(1, -1)                    # every code, scale 1
    q = F.quantize(vals, F.QuantConfig("mxfp4", block_size=16, allow_mse_scale=False))
    assert torch.equal(F.dequantize(q), vals)                     # OCP scale lands on s=1 here
    planes = F.to_digit_planes(q)
    assert len(planes) == 2
    d0, d1 = planes[0].float(), planes[1].float()
    assert set(d1.unique().tolist()) <= {-1., 0., 1.}, "2nd plane not ternary"
    aligned = q.codes.float()
    # balanced-digit support: d1 != 0 iff aligned outside [-8,7] -> {+8, +12, -12}; -8 fits
    # plane0 (theory SS1's "+-4/+-6 codes" is symmetric-imprecise; density is slightly lower)
    assert torch.equal((d1 != 0), (aligned > 7) | (aligned < -8)), "2nd plane support wrong"
    assert torch.equal(d0 + 16 * d1, aligned), "plane recomposition broken"


def test_int_formats_single_plane():
    x = _rand()
    for fid in ["int4_pc", "int4_g128", "mxint4", "e1m2", "nvint4"]:
        q = F.quantize(x, F.QuantConfig(fid))
        assert len(F.to_digit_planes(q)) == 1, fid
        assert F.estimate_cost(q)["stored_plane"] == 1, fid


def test_uint4_unsigned():
    x = torch.rand(8, 64) * 5                                     # nonnegative (softmax-P-like)
    q = F.quantize(x, F.QuantConfig("uint4"))
    assert q.codes.min() >= 0 and q.codes.max() <= 15
    planes = F.to_digit_planes(q)
    assert len(planes) == 1                                       # 1 unsigned digit, 2x resolution
    xb = x.reshape(8, 2, 32)
    assert ((xb.abs().amax(-1) / q.scales) < 16).all()            # int_bits = w*nD = 4
    xneg = x.clone(); xneg[0, 0] = -1.0                           # negatives clip to 0
    yneg = F.fake_quant_ste(xneg, F.QuantConfig("uint4"))
    assert yneg[0, 0] == 0


def test_nf4_r1_only():
    x = _rand()
    q = F.quantize(x, F.QuantConfig("nf4"))
    assert (F.dequantize(q) - x).pow(2).mean() / x.pow(2).mean() < 0.02
    try:
        F.to_digit_planes(q)
        assert False, "nf4 must refuse digit planes"
    except ValueError:
        pass


# ---------------------------------------------------------------- mixfp4 (per-block argmin)

def test_mixfp4_argmin():
    """MixFP4 kappa law (kappa*=2.2243): uniform blocks (kappa~1.7) -> INT grid;
    heavy-tailed blocks (gauss^3, kappa~3.1) -> E2M1. Measured 0.10 vs 0.76 e2m1-ratio."""
    torch.manual_seed(1)
    flat = torch.rand(256, 16) * 2 - 1                            # low kappa -> INT grid wins
    heavy = torch.randn(256, 16) ** 3                             # high kappa -> E2M1 wins
    x = torch.cat([flat, heavy], dim=0)
    cfg = F.QuantConfig("mixfp4")
    q = F.quantize(x, cfg)
    choice = q.plan.fmt_choice                                    # True = e2m1
    assert choice[256:].float().mean() > 0.6, "heavy-tailed blocks should pick E2M1"
    assert choice[:256].float().mean() < 0.25, "flat blocks should pick INT grid"
    err_mix = (F.dequantize(q) - x).pow(2).sum()
    for fid in ["e2m1_16", "e1m2"]:
        err = (F.fake_quant_ste(x, F.QuantConfig(fid)) - x).pow(2).sum()
        assert err_mix <= err + 1e-6, f"mix worse than uniform {fid}"
    cost = F.estimate_cost(q)
    assert cost["stored_plane"] == 2 and "e2m1_block_ratio" in cost


# ---------------------------------------------------------------- edge cases

def test_block_remainder():
    x = _rand(8, 100)                                             # 100 = 3x32 + 4
    for fid in ["mxint4", "mxfp4", "int4_g128"]:
        y = F.fake_quant_ste(x, F.QuantConfig(fid))
        assert y.shape == x.shape
        assert (y - x).pow(2).mean() / x.pow(2).mean() < 0.05, fid


def test_nan_raises():
    x = _rand(); x[3, 7] = float("nan")
    try:
        F.quantize(x, F.QuantConfig("mxint4"))
        assert False, "NaN must be a hard error"
    except ValueError:
        pass


def test_ste_gradient_identity():
    x = _rand().requires_grad_(True)
    for fid in ["mxint4", "mxfp4", "mixfp4", "nvfp4"]:
        F.fake_quant_ste(x, F.QuantConfig(fid)).sum().backward()
        assert torch.equal(x.grad, torch.ones_like(x)), fid
        x.grad = None


def test_gptq_two_phase_column_path():
    """find_params on a group slice, then per-column quantize == whole-slice quantize."""
    W = _rand(32, 32)
    for fid in ["mxint4", "mxfp4", "mixfp4", "int4_pc"]:
        cfg = F.QuantConfig(fid, block_size=32 if fid != "int4_pc" else None)
        plan = F.compute_scales(W, cfg)
        whole = F.dequantize(F.quantize_with_plan(W, plan))
        cols = torch.cat([F.dequantize(F.quantize_with_plan(W[:, i:i+1], plan))
                          for i in range(32)], dim=1)
        assert torch.equal(whole, cols), fid


def test_estimate_cost_beff():
    x = _rand(8, 128)
    for fid, beff in [("int4_g128", 4.125), ("mxint4", 4.25), ("mxfp4", 4.25),
                      ("nvfp4", 4.5), ("mixfp4", 4.5), ("nvint4", 4.5)]:
        got = F.estimate_cost(F.quantize(x, F.QuantConfig(fid)))["b_eff"]
        assert abs(got - beff) < 1e-9, f"{fid}: b_eff {got} != {beff}"


def test_block_stats():
    st = F.block_stats(_rand(), F.QuantConfig("mxint4"), candidates=("mxint4", "mxfp4"))
    assert (st["kappa"] >= 1 - 1e-4).all()
    assert "mse_mxint4" in st and "mse_mxfp4" in st and torch.isfinite(st["mse_mxfp4"]).all()


# ---------------------------------------------------------------- FlatQuant plugin (W14)

def test_flatquant_activation_quantizer_fmt():
    from methods.flatquant.flatquant.quant_utils import ActivationQuantizer
    aq = ActivationQuantizer(bits=4, sym=True, lac=True, fmt_cfg=F.QuantConfig("mxint4", role="a"))
    x = torch.randn(2, 8, 64, requires_grad=True)
    y = aq(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    # alpha rule: lac clip factors receive gradient through the tensor clamp
    assert aq.clip_factor_a_max.grad is not None and aq.clip_factor_a_max.grad.abs().sum() > 0


def test_flatquant_weight_quantizer_fmt():
    from methods.flatquant.flatquant.quant_utils import WeightQuantizer
    for fid in ["mxfp4", "mxint4"]:
        wq = WeightQuantizer()
        wq.configure(4, groupsize=-1, sym=True, fmt_cfg=F.QuantConfig(fid, role="w"))
        W = torch.randn(48, 64)
        wq.find_params(W)
        assert wq.ready()
        y = wq.quantize(W)
        assert y.shape == W.shape
        ref = F.fake_quant_ste_with_plan(W, wq._fmt_plan)
        assert torch.equal(y, ref), fid


def test_flatquant_fmt_bridge_parse():
    from methods.flatquant.flatquant import fmt_bridge
    cfg, pools = fmt_bridge.load_fmt_config(
        '{"a": {"fmt": "mxint4"}, "w": {"fmt": "mxfp4", "mse_scale": false}, "k": null}')
    assert cfg["a"].fmt_id == "mxint4" and cfg["w"].allow_mse_scale is False and "k" not in cfg
    assert pools == {}
    try:
        fmt_bridge.load_fmt_config('{"a": {"fmt": "no_such_fmt"}}')
        assert False
    except ValueError:
        pass


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL {name}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
