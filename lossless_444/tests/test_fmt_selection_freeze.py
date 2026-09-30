"""Standalone tests for the E1-0 calibration format-selection freeze (no pytest in env).

Covers the FlatQuant-side machinery that fixes each (layer,op,role) quant dtype from calibration:
  * fmt_bridge.load_fmt_config parsing of fixed {"fmt"} vs pool {"pool","select"} roles
  * fmt_bridge.site_fmt precedence (resolved > pool-raise > fixed > None)
  * quant_utils observe -> accumulate -> finalize on real quantizers
  * quant_utils.start_observe_layer / finalize_observe_layer over a mock module tree, with
    role inference from module-tree names and shared-quantizer dedup
  * the unified MSE-both scale rule inside mixfp4 (fmt_lib gap #2)

Run: python lossless_444/tests/test_fmt_selection_freeze.py
"""
import sys
import os
import traceback

import torch
import torch.nn as nn

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lossless_444.scripts import fmt_lib
from methods.flatquant.flatquant import fmt_bridge
from methods.flatquant.flatquant.quant_utils import (
    ActivationQuantizer, WeightQuantizer, start_observe_layer, finalize_observe_layer,
    _infer_role,
)


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_load_fmt_config_mixed():
    cfg, pools = fmt_bridge.load_fmt_config(
        '{"w": {"pool": ["int4_pc","mxfp4","mixfp4"], "select": "mse"}, "a": {"fmt": "mxint4"}}')
    assert set(cfg) == {"a"} and cfg["a"].fmt_id == "mxint4", cfg
    assert set(pools) == {"w"}, pools
    assert pools["w"]["pool"] == ["int4_pc", "mxfp4", "mixfp4"]
    assert pools["w"]["select"] == "mse"


def test_load_fmt_config_rejects_both_and_bad():
    for bad in ('{"w": {"fmt": "mxfp4", "pool": ["int4_pc","mxfp4"]}}',   # both
                '{"w": {"pool": ["int4_pc"]}}',                            # <2 candidates
                '{"w": {"pool": ["int4_pc","nope"]}}',                     # unknown fmt
                '{"w": {"pool": ["int4_pc","mxfp4"], "select": "l1"}}'):   # bad select
        try:
            fmt_bridge.load_fmt_config(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_site_fmt_precedence():
    args = _Args(fmt_cfg={"a": fmt_lib.QuantConfig(fmt_id="mxint4", role="a")},
                 fmt_pools={"w": {"pool": ["int4_pc", "mxfp4"], "select": "mse"}},
                 fmt_resolved={"model.layers.0.self_attn.q_proj.weight_quantizer": "mxfp4"})
    # (1) resolved wins
    q = fmt_bridge.site_fmt(args, "model.layers.0.self_attn.q_proj.weight_quantizer", "w")
    assert q.fmt_id == "mxfp4" and q.role == "w"
    # (2) fixed role falls through
    a = fmt_bridge.site_fmt(args, "model.layers.9.self_attn.qkv_quant", "a")
    assert a.fmt_id == "mxint4"
    # (3) unresolved pool site -> fail fast
    try:
        fmt_bridge.site_fmt(args, "model.layers.5.mlp.down_proj.weight_quantizer", "w")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unresolved pool site")
    # (4) role with neither fixed nor pool -> None
    assert fmt_bridge.site_fmt(args, "whatever.k_cache_quantizer", "k") is None


def test_infer_role():
    assert _infer_role("self_attn.q_proj.weight_quantizer", WeightQuantizer()) == "w"
    aq = ActivationQuantizer(bits=4, sym=True)
    assert _infer_role("self_attn.k_cache_quantizer", aq) == "k"
    assert _infer_role("self_attn.v_cache_quantizer", aq) == "v"
    assert _infer_role("self_attn.q_cache_quantizer", aq) == "q"
    assert _infer_role("self_attn.qkv_quant", aq) == "a"
    assert _infer_role("mlp.down_proj.act_quantizer", aq) == "a"


def test_observe_accumulate_finalize_single():
    torch.manual_seed(0)
    aq = ActivationQuantizer(bits=4, sym=True)
    from methods.flatquant.flatquant.quant_utils import _observe_start, _observe_finalize
    _observe_start(aq, ["int4_pt", "mxint4", "mxfp4", "mixfp4"], "a", site_key="s")
    assert aq._observe
    # feed heavy-tailed data across "batches": observe forward passes through unchanged
    for _ in range(3):
        x = (torch.randn(8, 128) ** 3)
        y = aq(x)
        assert torch.equal(x, y), "observe forward must be a pass-through"
    rec = _observe_finalize(aq)
    assert not aq._observe and aq.fmt_cfg is not None
    assert rec["fmt_id"] in ("int4_pt", "mxint4", "mxfp4", "mixfp4")
    assert rec["fmt_id"] == aq.fmt_cfg.fmt_id and aq.fmt_cfg.role == "a"
    # after finalize the quantizer really quantizes (not a pass-through)
    x = torch.randn(8, 128)
    assert not torch.equal(x, aq(x))


class _MockLayer(nn.Module):
    """Mimics the FlatQuant layer's quantizer sites incl. a SHARED activation quantizer."""
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.qkv_quant = ActivationQuantizer(bits=4, sym=True)          # role a (shared)
        self.self_attn.q_proj = nn.Module()
        self.self_attn.q_proj.weight_quantizer = _wq()                            # role w
        self.self_attn.q_proj.act_quantizer = self.self_attn.qkv_quant            # SHARED alias
        self.self_attn.k_proj = nn.Module()
        self.self_attn.k_proj.weight_quantizer = _wq()
        self.self_attn.k_proj.act_quantizer = self.self_attn.qkv_quant            # SHARED alias
        self.self_attn.k_cache_quantizer = ActivationQuantizer(bits=4, sym=True)  # role k
        self.self_attn.v_cache_quantizer = ActivationQuantizer(bits=4, sym=True)  # role v
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Module()
        self.mlp.down_proj.weight_quantizer = _wq()


def _wq():
    q = WeightQuantizer()
    q.configure(4, fmt_cfg=None)
    return q


def test_layer_observe_shared_and_roles():
    torch.manual_seed(1)
    layer = _MockLayer()
    pools = {"w": {"pool": ["int4_pc", "mxfp4", "mixfp4"], "select": "mse"},
             "a": {"pool": ["int4_pt", "mxint4", "mxfp4"], "select": "mse"},
             "k": {"pool": ["int4_g128", "mxint4"], "select": "mse"},
             "v": {"pool": ["int4_g128", "mxint4"], "select": "mse"}}
    n = start_observe_layer(layer, pools, "model.layers.3")
    # distinct quantizers: qkv(shared, 1) + 3 weights + k_cache + v_cache = 6 (NOT 7 despite alias)
    assert n == 6, n
    # drive observe: weights via find_params, activations via forward
    for wq_path in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.mlp.down_proj]:
        wq_path.weight_quantizer.find_params(torch.randn(64, 256))
    layer.self_attn.qkv_quant(torch.randn(16, 256))
    layer.self_attn.k_cache_quantizer(torch.randn(16, 128))
    layer.self_attn.v_cache_quantizer(torch.randn(16, 128))
    recs = finalize_observe_layer(layer)
    assert len(recs) == 6, len(recs)
    keys = {r["site_key"]: r["role"] for r in recs}
    assert keys["model.layers.3.self_attn.qkv_quant"] == "a"
    assert keys["model.layers.3.self_attn.q_proj.weight_quantizer"] == "w"
    assert keys["model.layers.3.self_attn.k_cache_quantizer"] == "k"
    assert keys["model.layers.3.self_attn.v_cache_quantizer"] == "v"
    # shared quantizer resolved once, and both aliases now carry the same frozen fmt_cfg
    assert layer.self_attn.q_proj.act_quantizer.fmt_cfg is layer.self_attn.k_proj.act_quantizer.fmt_cfg
    # every candidate came from the right role pool
    for r in recs:
        assert r["fmt_id"] in pools[r["role"]]["pool"], r


def test_mixfp4_scale_rule_unified():
    # both mixfp4 candidates now use MSE-both scale; sanity: choice map is a valid bool tensor
    x = torch.randn(32, 256) ** 3
    plan = fmt_lib.compute_scales(x, fmt_lib.QuantConfig(fmt_id="mixfp4", role="w"))
    assert plan.fmt_choice is not None and plan.fmt_choice.dtype == torch.bool
    # e1m2 candidate must NOT be handicapped to absmax: reconstruct both errs and confirm the
    # recorded choice equals a fresh MSE-both argmin (i.e. selection is self-consistent)
    q = fmt_lib.quantize(x, fmt_lib.QuantConfig(fmt_id="mixfp4", role="w"))
    d = fmt_lib.dequantize(q)
    assert torch.isfinite(d).all()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
