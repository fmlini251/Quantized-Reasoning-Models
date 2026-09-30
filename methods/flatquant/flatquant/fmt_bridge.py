"""fmt_bridge -- W14/E1-0: plug lossless_444/scripts/fmt_lib.py formats into FlatQuant.

Parses --fmt_config (JSON path or inline JSON) into per-role directives and hands them to the
quantizers.  Roles: w / a / q / k / v.  A role entry is one of:

  * FIXED grid  -- {"fmt": "mxfp4", ...}            -> every site of the role uses that format
  * CALIB pool  -- {"pool": [...], "select": "mse"} -> the format is CHOSEN per (layer,op,role)
                                                       from calibration and frozen (E1-0)

    {"w": {"pool": ["int4_pc","mxint4","mxfp4","mixfp4"], "select": "mse"},
     "a": {"fmt": "mxint4"}, "k": {"fmt": "mxint4"}, "v": {"fmt": "mxint4"}}

A role absent (or null) keeps the original FlatQuant uniform INT quantizer.  Fixed roles ->
args.fmt_cfg[role] (QuantConfig); pool roles -> args.fmt_pools[role] ({"pool","select"}).
The per-site frozen choices are written into args.fmt_resolved {site_key: fmt_id} during the
calibration observe pass (train_utils.cali_flat_quant) and consulted via site_fmt().

NOTE: bits gating stays with --w_bits/--a_bits/... -- a role with a fmt/pool entry must still be
launched with the corresponding bits < 16, otherwise the quantizer is a pass-through and the
entry would silently do nothing (we raise instead, fail-fast).
"""
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lossless_444.scripts import fmt_lib  # noqa: E402

ROLES = ("w", "a", "q", "k", "v")
_SELECT_METHODS = ("mse", "absmax")


def load_fmt_config(path_or_json):
    """--fmt_config value -> (fmt_cfg, fmt_pools).

    fmt_cfg   = {role: fmt_lib.QuantConfig}  for FIXED-grid roles
    fmt_pools = {role: {"pool": [...], "select": "mse"}}  for CALIB-selected roles
    """
    if not path_or_json:
        return {}, {}
    if os.path.exists(path_or_json):
        with open(path_or_json) as f:
            raw = json.load(f)
    else:
        raw = json.loads(path_or_json)
    cfg, pools = {}, {}
    for role, entry in raw.items():
        if role not in ROLES:
            raise ValueError(f"--fmt_config: unknown role '{role}' (expected one of {ROLES})")
        if entry is None:
            continue
        if ("fmt" in entry) == ("pool" in entry):
            raise ValueError(f"--fmt_config role '{role}': give exactly one of 'fmt' or 'pool'")
        if "pool" in entry:
            pool = list(entry["pool"])
            if len(pool) < 2:
                raise ValueError(f"--fmt_config role '{role}': 'pool' needs >=2 candidate formats")
            unknown = [f for f in pool if f not in fmt_lib.FORMATS]
            if unknown:
                raise ValueError(f"--fmt_config role '{role}': unknown fmt(s) {unknown} "
                                 f"(known: {sorted(fmt_lib.FORMATS)})")
            select = entry.get("select", "mse")
            if select not in _SELECT_METHODS:
                raise ValueError(f"--fmt_config role '{role}': select must be one of "
                                 f"{_SELECT_METHODS}, got {select!r}")
            pools[role] = {"pool": pool, "select": select}
        else:
            fmt_id = entry["fmt"]
            if fmt_id not in fmt_lib.FORMATS:
                raise ValueError(f"--fmt_config: unknown fmt '{fmt_id}' "
                                 f"(known: {sorted(fmt_lib.FORMATS)})")
            cfg[role] = fmt_lib.QuantConfig(
                fmt_id=fmt_id,
                block_size=entry.get("block_size"),
                scale_dtype=entry.get("scale_dtype"),
                allow_mse_scale=entry.get("mse_scale"),
                rounding=entry.get("rounding", "rne"),
                role=role,
            )
    return cfg, pools


def validate_bits(args):
    """Fail fast if a fmt/pool role is launched with bits==16 (quantizer would pass through)."""
    bits = {"w": args.w_bits, "a": args.a_bits, "q": args.q_bits,
            "k": args.k_bits, "v": args.v_bits}
    roles = set(getattr(args, "fmt_cfg", {})) | set(getattr(args, "fmt_pools", {}))
    for role in roles:
        if bits[role] >= 16:
            raise ValueError(f"--fmt_config has role '{role}' but --{role}_bits is 16; "
                             f"set --{role}_bits 4 (bits gate enablement, fmt sets the grid)")


def role_fmt(args, role):
    """Fixed-grid QuantConfig for a role, or None (pool roles resolve per-site via site_fmt)."""
    return getattr(args, "fmt_cfg", {}).get(role)


def role_pool(args, role):
    """Candidate-pool directive {'pool','select'} for a role, or None."""
    return getattr(args, "fmt_pools", {}).get(role)


def site_fmt(args, site_key, role):
    """QuantConfig for a specific quantization site (layer/op/role).

    Precedence: (1) a per-site format frozen by the calibration observe pass
    (args.fmt_resolved[site_key]); (2) the role's fixed grid (role_fmt); (3) None (uniform).
    If the role is in pool (selection) mode but this site was never resolved, that is a bug in
    the freeze pass -- fail fast rather than silently fall back to uniform INT."""
    resolved = getattr(args, "fmt_resolved", {})
    if site_key in resolved:
        return fmt_lib.QuantConfig(fmt_id=resolved[site_key], role=role)
    if role_pool(args, role) is not None:
        raise ValueError(f"site '{site_key}' (role '{role}') is in pool-selection mode but was "
                         f"not resolved by the calibration freeze pass (fmt_resolved has "
                         f"{len(resolved)} entries)")
    return role_fmt(args, role)
