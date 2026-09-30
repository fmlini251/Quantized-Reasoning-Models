"""fmt_lib -- W1: canonical 4-bit format library for the lossless_444 mixed-precision experiments.

Single source of truth for every 4-bit storage format used in Track C / the mixed-precision
experiment plan (lossless_444/4bit_mixed_precision_experiment.md, mixed_4bit.md SS1).  Implements
the W1 API contract:

    quantize(x, cfg)        -> QuantizedTensor      (codes + scales + metadata)
    dequantize(q)           -> torch.Tensor
    fake_quant_ste(x, cfg)  -> torch.Tensor         (quant-dequant with straight-through gradient)
    compute_scales(x, cfg)  -> ScalePlan            (two-phase API for GPTQ: freeze scales, then
    quantize_with_plan(x, plan) -> QuantizedTensor   quantize columns against the frozen plan)
    to_digit_planes(q, w=4) -> list[torch.Tensor]   (ozaki balanced base-2^w signed peel)
    estimate_cost(q)        -> dict                 (b_eff + stored/executed plane + density, SS0.2)
    block_stats(x, cfg, candidates) -> dict         (kappa, kurtosis, per-format MSE -- C0/E1-0)

Conventions (mixed_4bit.md SS1.4):
  * Blocks run along the LAST dim ("rows x blocks"); callers reshape so that the last dim is the
    role's block axis (W/A: reduction dim k; K/V: head_dim).  block_size=-1 = one block per row.
  * rounding: round-to-nearest-even by default ("rne").  Note ozaki prealign itself rounds
    half-away-from-zero; this does NOT affect digit-collapse exactness -- once stored values sit on
    a pow2-aligned integer grid, the ozaki re-encode is exact under either rounding mode.
  * MXINT4 uses the production ozaki block-FP rule (s = 2^(frexp_exp(amax) - 3), clamp [-8,7],
    flash_ozaki/flash_oz1fp_codegen.py::_bfp_scale_torch) so that "MXINT4 == ozaki 1-digit encode"
    holds bit-exactly.
  * E2M1 grids (mxfp4/nvfp4) default to MSE-opt scale; INT grids default to absmax.
  * NaN/Inf in the input is a hard error (run fail, SS1.4).
  * nf4 is an R1 (fake-quant) control only: irrational grid, to_digit_planes() raises.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import torch

__all__ = [
    "FmtSpec", "QuantConfig", "ScalePlan", "QuantizedTensor", "FORMATS",
    "resolve", "quantize", "dequantize", "fake_quant_ste", "fake_quant_ste_with_plan",
    "compute_scales", "quantize_with_plan",
    "to_digit_planes", "estimate_cost", "block_stats", "format_selection_sse",
]

# --------------------------------------------------------------------------------------
# format registry
# --------------------------------------------------------------------------------------

# E2M1 positive levels; aligned integers = level * 2 in {0,1,2,3,4,6,8,12}  (theory SS1: 2 digits)
_E2M1_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# QLoRA NF4 quantile table (asymmetric, irrational -- R1 control only)
_NF4_LEVELS = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)


@dataclass(frozen=True)
class FmtSpec:
    """Static description of one storage format (a row of mixed_4bit.md SS1.1/SS1.2)."""
    fmt_id: str
    kind: str                      # "int" (uniform grid) | "codebook" | "mix" | "nf4"
    qmin: int = 0                  # int kind: aligned-integer code range
    qmax: int = 0
    pos_levels: tuple = ()         # codebook kind: positive half (symmetric mirror)
    grid_mult: int = 1             # aligned_int = level_value * grid_mult (integer lattice snap)
    block_size: int = -1           # -1 = whole row
    scale_dtype: str = "fp16"      # fp16 | fp32 | e8m0 | e4m3
    tensor_scale: bool = False     # nvfp4: extra per-tensor fp32 scale
    mse_scale: bool = False        # default scale search mode (SS1.4)
    signed: bool = True
    format_bits: float = 0.0       # per-block metadata bits (mixfp4 Type-in-Scale = 0)
    scale_bits: int = 16
    mix_of: tuple = ()             # "mix" kind: candidate fmt_ids

    def _aligned_range(self) -> tuple:
        """(lo, hi) of the aligned-integer lattice."""
        if self.kind == "int":
            return self.qmin, self.qmax
        if self.kind == "codebook":
            hi = int(round(max(self.pos_levels) * self.grid_mult))
            return (-hi if self.signed else 0), hi
        if self.kind == "mix":
            los, his = zip(*(FORMATS[f]._aligned_range() for f in self.mix_of))
            return min(los), max(his)
        return 0, 0

    @property
    def max_aligned(self) -> int:
        lo, hi = self._aligned_range()
        return max(abs(lo), abs(hi))

    @property
    def gmax_level(self) -> float:
        """Largest representable level in level units (scale normalizer)."""
        if self.kind == "int":
            return float(self.qmax) / self.grid_mult
        if self.kind == "codebook":
            return float(max(self.pos_levels))
        if self.kind == "nf4":
            return 1.0
        raise ValueError(self.fmt_id)

    def stored_planes(self, w: int = 4) -> int:
        """Digit planes needed to hold the aligned integer grid exactly (theory SS1):
        signed nD planes cover [-2^(w*nD-1), 2^(w*nD-1)-1]; unsigned cover [0, 2^(w*nD)-1]."""
        if self.kind == "nf4":
            raise ValueError("nf4 has no integer lattice (R1 control only)")
        lo, hi = self._aligned_range()
        nd = 1
        while ((self.signed and not (lo >= -(1 << (w * nd - 1)) and hi <= (1 << (w * nd - 1)) - 1))
               or (not self.signed and hi > (1 << (w * nd)) - 1)):
            nd += 1
        return nd


FORMATS: dict[str, FmtSpec] = {s.fmt_id: s for s in [
    # -- standard formats (experiment group A, SS1.1) --
    FmtSpec("int4_pc",  "int", -8, 7, block_size=-1,  scale_dtype="fp16", scale_bits=16),
    FmtSpec("int4_pt",  "int", -8, 7, block_size=-1,  scale_dtype="fp16", scale_bits=16),
    FmtSpec("int4_g128","int", -8, 7, block_size=128, scale_dtype="fp16", scale_bits=16),
    FmtSpec("nvint4",   "int", -8, 7, block_size=16,  scale_dtype="e4m3", scale_bits=8),
    FmtSpec("mxint4",   "int", -8, 7, block_size=32,  scale_dtype="e8m0", scale_bits=8),
    FmtSpec("mxfp4",    "codebook", pos_levels=_E2M1_POS, grid_mult=2,
            block_size=32, scale_dtype="e8m0", scale_bits=8, mse_scale=True),
    FmtSpec("nvfp4",    "codebook", pos_levels=_E2M1_POS, grid_mult=2,
            block_size=16, scale_dtype="e4m3", scale_bits=8, tensor_scale=True, mse_scale=True),
    # E1M2 == INT grid x 1/2 (survey SS1): aligned ints [-7,7], mult 2
    FmtSpec("e1m2",     "int", -7, 7, grid_mult=2, block_size=16, scale_dtype="e4m3", scale_bits=8),
    FmtSpec("mixfp4",   "mix", mix_of=("e2m1_16", "e1m2"), block_size=16, scale_dtype="e4m3",
            scale_bits=8, format_bits=0.0),   # Type-in-Scale: format bit hides in scale sign
    FmtSpec("nf4",      "nf4", block_size=64, scale_dtype="fp32", scale_bits=32),
    # internal: e2m1 grid at block 16 / e4m3 (mixfp4 branch)
    FmtSpec("e2m1_16",  "codebook", pos_levels=_E2M1_POS, grid_mult=2,
            block_size=16, scale_dtype="e4m3", scale_bits=8, mse_scale=True),
    # -- custom formats (experiment group B, SS1.2) --
    FmtSpec("uint4",    "int", 0, 15, block_size=32, scale_dtype="e8m0", scale_bits=8, signed=False),
]}


@dataclass
class QuantConfig:
    """W1 contract: fmt_id, block_size, scale_dtype, axis, symmetric, allow_mse_scale."""
    fmt_id: str
    block_size: Optional[int] = None       # None -> format default
    scale_dtype: Optional[str] = None
    axis: str = "last"                     # blocks along last dim (callers reshape per role)
    symmetric: bool = True
    allow_mse_scale: Optional[bool] = None
    rounding: str = "rne"                  # "rne" | "half_away" (ozaki prealign convention)
    role: str = ""                         # informational: w/a/q/k/v/p (manifests, cost tables)


@dataclass
class ScalePlan:
    """Frozen scales (+ mixfp4 choice map) from compute_scales(); GPTQ quantizes against this."""
    spec: FmtSpec
    cfg: QuantConfig
    scales: torch.Tensor                    # (rows, nblocks) fp32, level units
    tensor_scale: Optional[torch.Tensor]    # nvfp4 per-tensor fp32 scalar
    fmt_choice: Optional[torch.Tensor]      # mixfp4: (rows, nblocks) bool, True = e2m1
    width: int                              # last-dim width the plan was computed for


@dataclass
class QuantizedTensor:
    """codes: aligned-integer lattice codes (int8) except nf4 (level index)."""
    codes: torch.Tensor
    scales: torch.Tensor
    plan: ScalePlan
    orig_shape: tuple
    orig_dtype: torch.dtype
    zero_points: Optional[torch.Tensor] = None
    metadata: dict = field(default_factory=dict)


def resolve(cfg: QuantConfig) -> FmtSpec:
    spec = FORMATS[cfg.fmt_id]
    if cfg.block_size is not None and cfg.block_size != spec.block_size:
        spec = replace(spec, block_size=cfg.block_size)
    if cfg.scale_dtype is not None and cfg.scale_dtype != spec.scale_dtype:
        spec = replace(spec, scale_dtype=cfg.scale_dtype)
    return spec


# --------------------------------------------------------------------------------------
# scale rules (theory SS2: where the scale lives decides the HW cost, not exactness)
# --------------------------------------------------------------------------------------

def _round(t: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "rne":
        return torch.round(t)
    if mode == "half_away":
        return torch.trunc(t + torch.where(t >= 0, 0.5, -0.5))
    raise ValueError(mode)


def _e4m3_round(s: torch.Tensor) -> torch.Tensor:
    """Round positive scales to the E4M3 (fn) grid, saturating at 448, floor at 2^-9."""
    if hasattr(torch, "float8_e4m3fn"):
        return s.clamp(min=2.0 ** -9, max=448.0).to(torch.float8_e4m3fn).to(torch.float32)
    e = torch.floor(torch.log2(s.clamp(min=2.0 ** -9, max=448.0)))
    step = torch.exp2(e - 3)                       # 3 mantissa bits
    return torch.round(s.clamp(min=2.0 ** -9, max=448.0) / step) * step


def _frexp_pow2_scale(amax: torch.Tensor, int_bits: int) -> torch.Tensor:
    """Production ozaki block-FP scale: 2^(frexp_exp(amax) - int_bits); amax/s in [2^(ib-1), 2^ib)."""
    _, e = torch.frexp(amax.clamp(min=1e-30))
    return torch.exp2((e - int_bits).to(torch.float32))


def _base_scale(amax: torch.Tensor, spec: FmtSpec) -> torch.Tensor:
    """absmax scale in level units (deq = level * s). amax==0 -> s=1 (canonical zero)."""
    g = spec.gmax_level
    if spec.scale_dtype == "e8m0":
        if spec.kind == "int":
            # int_bits: signed range [-2^ib, 2^ib-1] (mxint4: ib=3, == ozaki w*nD-1);
            # unsigned range [0, 2^ib-1] (uint4: ib=4, == ozaki unsigned split w*nD)
            ib = max((abs(spec.qmin) - 1).bit_length(), spec.qmax.bit_length()) \
                if spec.signed else spec.qmax.bit_length()
            s = _frexp_pow2_scale(amax, ib)        # mxint4: ib=3 == ozaki 1-digit encode
        else:
            # OCP MX rule for fp grids: 2^(floor(log2(amax)) - emax_elem); values saturate to gmax
            _, e = torch.frexp(amax.clamp(min=1e-30))
            emax = math.floor(math.log2(g))
            s = torch.exp2((e - 1 - emax).to(torch.float32))
    else:
        s = amax / g
    s = torch.where(amax == 0, torch.ones_like(s), s)
    if spec.scale_dtype == "fp16":
        s = s.clamp(min=6e-5).to(torch.float16).to(torch.float32)
    elif spec.scale_dtype == "e4m3":
        s = _e4m3_round(s)
    return s


def _snap_codes(t: torch.Tensor, spec: FmtSpec, rounding: str) -> torch.Tensor:
    """Level-unit values -> aligned integer codes on the format lattice."""
    if spec.kind == "int":
        q = _round(t * spec.grid_mult, rounding) / spec.grid_mult   # grid step 1/mult in level units
        return (q.clamp(spec.qmin / spec.grid_mult if spec.signed else 0,
                        spec.qmax / spec.grid_mult) * spec.grid_mult).round()
    if spec.kind == "codebook":
        lv = torch.tensor(spec.pos_levels, dtype=t.dtype, device=t.device)
        mid = (lv[1:] + lv[:-1]) / 2
        idx = torch.bucketize(t.abs(), mid)
        return (lv[idx] * torch.sign(t) * spec.grid_mult).round()
    raise ValueError(spec.kind)


def _mse_scale_search(xb: torch.Tensor, s0: torch.Tensor, spec: FmtSpec, rounding: str,
                      pow2: bool) -> torch.Tensor:
    """Per-block scale search minimizing MSE (SS1.4: E2M1 grids default MSE-opt)."""
    if pow2:
        cands = [s0 * (2.0 ** j) for j in (-1, 0, 1)]
    else:
        cands = [s0 * p for p in torch.linspace(0.55, 1.0, 10).tolist()]
    best_err = torch.full_like(s0, float("inf"))
    best_s = s0.clone()
    for s in cands:
        s = torch.where(s <= 0, torch.ones_like(s), s)
        if spec.scale_dtype == "e4m3":
            s = _e4m3_round(s)
        codes = _snap_codes(xb / s.unsqueeze(-1), spec, rounding)
        err = ((codes / spec.grid_mult) * s.unsqueeze(-1) - xb).pow(2).sum(-1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, s, best_s)
    return best_s


# --------------------------------------------------------------------------------------
# core quantize / dequantize
# --------------------------------------------------------------------------------------

def _to_blocks(x: torch.Tensor, block: int):
    """(rows, K) -> (rows, nblocks, B) zero-padded; remainder block keeps its true amax (SS1.4)."""
    rows, K = x.shape
    B = K if block == -1 else block
    nb = (K + B - 1) // B
    pad = nb * B - K
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x.reshape(rows, nb, B), B, pad


def compute_scales(x: torch.Tensor, cfg: QuantConfig) -> ScalePlan:
    """Phase 1: block scales (+ mixfp4 format map) from a full-width tensor. GPTQ freezes this."""
    spec = resolve(cfg)
    if not torch.isfinite(x).all():
        raise ValueError(f"NaN/Inf entering quantize ({cfg.fmt_id}, role={cfg.role})")
    xf = x.detach().reshape(-1, x.shape[-1]).float()
    if spec.kind == "mix":
        # Unified scale rule (4bit_mixed_precision_experiment.md gap #2): both candidates get
        # their per-block MSE-optimal scale ("each format at its best") BEFORE the argmin, so the
        # per-block format choice reflects the grid, not an asymmetric scale handicap. Previously
        # e2m1_16 used mse_scale=True but e1m2 (int grid) defaulted to absmax, biasing the argmin
        # toward e2m1. allow_mse_scale=True forces MSE-both.
        plans = [compute_scales(xf, replace(cfg, fmt_id=f, block_size=spec.block_size,
                                            allow_mse_scale=True))
                 for f in spec.mix_of]
        errs = []
        for p in plans:
            q = quantize_with_plan(xf, p)
            xb, _, _ = _to_blocks(xf, p.spec.block_size)
            db, _, _ = _to_blocks(_dequant_2d(q), p.spec.block_size)
            errs.append((db - xb).pow(2).sum(-1))
        choice = errs[0] <= errs[1]                             # True = first candidate (e2m1)
        scales = torch.where(choice, plans[0].scales, plans[1].scales)
        return ScalePlan(spec, cfg, scales, None, choice, xf.shape[-1])
    if not spec.signed and (xf < 0).any():
        xf = xf.clamp(min=0)                                    # unsigned fmt: negatives clip to 0
    xb, B, _ = _to_blocks(xf, spec.block_size)
    amax = xb.abs().amax(-1)
    ts = None
    if spec.tensor_scale:                                       # nvfp4: fp32 tensor scale first
        ts = (xf.abs().max() / (448.0 * spec.gmax_level)).clamp(min=1e-30)
        xb = xb / ts
        amax = amax / ts
    s = _base_scale(amax, spec)
    use_mse = spec.mse_scale if cfg.allow_mse_scale is None else cfg.allow_mse_scale
    if use_mse and spec.kind in ("int", "codebook"):
        s = _mse_scale_search(xb, s, spec, cfg.rounding, pow2=(spec.scale_dtype == "e8m0"))
    return ScalePlan(spec, cfg, s, ts, None, xf.shape[-1])


def quantize_with_plan(x: torch.Tensor, plan: ScalePlan) -> QuantizedTensor:
    """Phase 2: snap x (full width, or a column slice iff nblocks==1) to the frozen plan."""
    spec, cfg = plan.spec, plan.cfg
    orig_shape, orig_dtype = x.shape, x.dtype
    xf = x.detach().reshape(-1, x.shape[-1]).float()
    if xf.shape[-1] != plan.width and plan.scales.shape[-1] != 1:
        raise ValueError("partial-width quantize needs a single-block plan (GPTQ column path)")
    if plan.tensor_scale is not None:
        xf = xf / plan.tensor_scale
    if not spec.signed:
        xf = xf.clamp(min=0)
    if spec.kind == "mix":
        specs = [FORMATS[f] for f in spec.mix_of]
        if xf.shape[-1] == plan.width:
            xb, _, _ = _to_blocks(xf, spec.block_size)
        else:
            xb = xf.unsqueeze(1)                    # GPTQ column slice vs single-block plan
        t = xb / plan.scales.unsqueeze(-1)
        c0 = _snap_codes(t, specs[0], cfg.rounding)
        c1 = _snap_codes(t, specs[1], cfg.rounding)
        codes = torch.where(plan.fmt_choice.unsqueeze(-1), c0, c1)
    elif spec.kind == "nf4":
        xb, B, _ = _to_blocks(xf, spec.block_size)
        lv = torch.tensor(_NF4_LEVELS, dtype=xb.dtype, device=xb.device)
        mid = (lv[1:] + lv[:-1]) / 2
        codes = torch.bucketize(xb / plan.scales.unsqueeze(-1), mid).to(torch.int16)
    else:
        if xf.shape[-1] == plan.width:
            xb, _, _ = _to_blocks(xf, spec.block_size)
        else:
            xb = xf.unsqueeze(1)                    # GPTQ column slice vs single-block plan
        codes = _snap_codes(xb / plan.scales.unsqueeze(-1), spec, cfg.rounding)
    if spec.kind != "nf4":
        codes = codes.to(torch.int16) if spec.max_aligned > 127 else codes.to(torch.int8)
    return QuantizedTensor(codes, plan.scales, plan, orig_shape, orig_dtype)


def quantize(x: torch.Tensor, cfg: QuantConfig) -> QuantizedTensor:
    return quantize_with_plan(x, compute_scales(x, cfg))


def _dequant_2d(q: QuantizedTensor) -> torch.Tensor:
    spec, plan = q.plan.spec, q.plan
    if spec.kind == "nf4":
        lv = torch.tensor(_NF4_LEVELS, dtype=torch.float32, device=q.codes.device)
        vals = lv[q.codes.long()] * q.scales.unsqueeze(-1)
    else:
        mult = FORMATS[spec.mix_of[0]].grid_mult if spec.kind == "mix" else spec.grid_mult
        vals = (q.codes.float() / mult) * q.scales.unsqueeze(-1)
    if plan.tensor_scale is not None:
        vals = vals * plan.tensor_scale
    rows = vals.shape[0]
    return vals.reshape(rows, -1)


def dequantize(q: QuantizedTensor) -> torch.Tensor:
    flat = _dequant_2d(q)
    K = q.orig_shape[-1]
    return flat[:, :K].reshape(q.orig_shape).to(q.orig_dtype)


def fake_quant_ste(x: torch.Tensor, cfg: QuantConfig) -> torch.Tensor:
    """Quant-dequant with straight-through gradient (identity backward).

    Clipping (FlatQuant lac/lwc) must be applied to the TENSOR before this call -- the alpha
    double-clipping rule of 4bit_mixed_precision_experiment.md SS3.2: fmt_lib receives the
    already-clipped tensor and owns only the grid + scale."""
    y = dequantize(quantize(x, cfg))
    return x + (y - x).detach()


def fake_quant_ste_with_plan(x: torch.Tensor, plan: ScalePlan) -> torch.Tensor:
    """STE quant-dequant against a frozen ScalePlan (WeightQuantizer/GPTQ two-phase path)."""
    y = dequantize(quantize_with_plan(x, plan))
    return x + (y - x).detach()


# --------------------------------------------------------------------------------------
# digit planes / cost / stats
# --------------------------------------------------------------------------------------

def to_digit_planes(q: QuantizedTensor, w: int = 4) -> list:
    """Aligned-int codes -> ozaki place-value signed digit planes (low->high balanced peel,
    same convention as flash_oz1fp_codegen._digit_planes). Unsigned formats use the unsigned
    split (int_bits = w*nD, W8/theory SS7)."""
    spec = q.plan.spec
    if spec.kind == "nf4":
        raise ValueError("nf4 has no integer lattice; storage-only control (theory SS1)")
    signed = spec.signed
    nd = spec.stored_planes(w)
    base, half = 1 << w, 1 << (w - 1)
    planes, cur = [], q.codes.long().clone()
    for t in range(nd):
        if t == nd - 1:
            d = cur
        elif signed:
            lo = cur & (base - 1)
            d = torch.where(lo >= half, lo - base, lo)
        else:
            d = cur & (base - 1)
        planes.append(d.to(torch.int8))
        cur = (cur - d) >> w
    return planes


def estimate_cost(q: QuantizedTensor, w: int = 4) -> dict:
    """SS0.2 accounting: b_eff = 4 + meta_bits/block; plane recorded 3 ways."""
    spec = q.plan.spec
    B = spec.block_size if spec.block_size != -1 else q.orig_shape[-1]
    b_eff = 4.0 + (spec.scale_bits + spec.format_bits) / B
    out = {"fmt_id": spec.fmt_id, "b_eff": b_eff, "block_size": B,
           "scale_dtype": spec.scale_dtype, "tensor_scale": spec.tensor_scale}
    if spec.kind == "nf4":
        out.update(stored_plane=None, executed_plane=None, nonzero_density=None,
                   note="no integer lattice (R1 control)")
        return out
    planes = to_digit_planes(q, w)
    dens = [float((p != 0).float().mean()) for p in planes[1:]]
    if spec.kind == "mix" and q.plan.fmt_choice is not None:
        out["e2m1_block_ratio"] = float(q.plan.fmt_choice.float().mean())
    out.update(stored_plane=len(planes), executed_plane=len(planes),
               nonzero_density=dens or [0.0])
    return out


def block_stats(x: torch.Tensor, cfg: QuantConfig, candidates: tuple = ()) -> dict:
    """Per-block kappa=amax/RMS, kurtosis, and per-candidate-format MSE (C0/E1-0 dumps)."""
    spec = resolve(cfg)
    xf = x.detach().reshape(-1, x.shape[-1]).float()
    xb, B, pad = _to_blocks(xf, spec.block_size)
    n = xb.shape[-1] - (pad if pad else 0)
    amax = xb.abs().amax(-1)
    rms = xb.pow(2).sum(-1).div(max(n, 1)).sqrt().clamp(min=1e-30)
    mu = xb.sum(-1, keepdim=True) / max(n, 1)
    var = (xb - mu).pow(2).sum(-1).div(max(n, 1)).clamp(min=1e-30)
    kurt = (xb - mu).pow(4).sum(-1).div(max(n, 1)) / var.pow(2)
    out = {"kappa": amax / rms, "kurtosis": kurt, "amax": amax, "rms": rms}
    for fid in candidates:
        # unified scale rule (MSE-both) so per-format MSE is a fair grid comparison (gap #2)
        c = replace(cfg, fmt_id=fid, block_size=None, scale_dtype=None, allow_mse_scale=True)
        d = fake_quant_ste(xf, c)
        db, _, _ = _to_blocks(d.detach(), spec.block_size)
        out[f"mse_{fid}"] = (db - xb).pow(2).sum(-1) / max(n, 1)
    return out


def format_selection_sse(x: torch.Tensor, pool, role: str = "",
                         block_size: Optional[int] = None, scale_dtype: Optional[str] = None,
                         unify_scale: str = "mse") -> dict:
    """Per-candidate total SSE for one tensor, under a UNIFIED scale rule (E1-0 format-selection
    freeze; 4bit_mixed_precision_experiment.md SS3.4).  The whole-tensor SSE (summed over every
    block/row) is the score a per-(layer,op,role) argmin uses to pick ONE format for the site.

    unify_scale="mse" (default): every candidate gets its own per-block MSE-optimal scale first
    ("each format at its best" -- the fair grid comparison the user selected).  "absmax": every
    candidate uses the plain absmax/microscaling scale (paper-Algorithm-1 fidelity).

    Returns {fmt_id: sse_float, "_energy": sum(x^2)}; callers accumulate across the calibration
    set and argmin at the end.  Grad-free (no STE): this runs inside torch.no_grad() observe."""
    if unify_scale == "mse":
        allow = True
    elif unify_scale == "absmax":
        allow = False
    else:
        raise ValueError(f"unify_scale must be 'mse' or 'absmax', got {unify_scale!r}")
    xf = x.detach().reshape(-1, x.shape[-1]).float()
    out = {"_energy": float(xf.pow(2).sum())}
    for fid in pool:
        cfg = QuantConfig(fmt_id=fid, block_size=block_size, scale_dtype=scale_dtype,
                          allow_mse_scale=allow, role=role)
        d = dequantize(quantize(xf, cfg)).float()
        out[fid] = float((d - xf).pow(2).sum())
    return out
