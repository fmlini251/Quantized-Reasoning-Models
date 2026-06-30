#!/usr/bin/env python3
"""Sweep the Linear-nmp x Attention-nmp accuracy grid for ozaki1_fp (w=4) on MATH-500.

Fills this 2-D table (mirrors the layout the user asked about):

    Linear \\ Attn   1     3     4     6     9     10    BF16
    1               .     .     .     .     .     .     .
    3               .     .     .     .     .     .     .
    ...
    BF16            .     .     .     .     .     .     0.948

    rows  = nmp of the LINEAR layers (qkv_proj / o_proj / gate_up_proj / down_proj)
    cols  = nmp of the ATTENTION score matmuls (QK^T / PV)
    cell  = MATH-500 extractive_match accuracy
    BF16  = that operation stays native (un-emulated)

Each cell is ONE `inference_vllm.py` run. The cell -> flags mapping (verified bit-for-bit
against the existing runs in outputs/inference/):

    (L,    BF16)        linear_only  --nmp L                            attn native
    (BF16, A)           attn_only    --nmp A                            linear native
    (BF16, BF16)        off                                            fully native (= 0.948)
    (L, A)  L == A      full         --nmp L                            both at L
    (L, A)  L != A      full         --nmp A  --nmp_overrides <lin>=L   attn=A (base nmp),
                                                                        linear=L (per-op override)

The attention backend reads the *base* --nmp; the per-op --nmp_overrides only match the linear
layer names (qkv_proj/o_proj/gate_up_proj/down_proj). So setting the base nmp to A and overriding
every linear name to L gives linear=L, attn=A in one `full` run -- exactly how the existing
(L=1, A=6)=0.426 cell was produced. Base config mirrors configs/ozaki1_fp_per_op_gsm8k.yaml:
ozaki1_fp / gemm_bits=4 / byte_split_style=all_signed_no_clamp / k=32 / weight_cache.

Sweep order & early-stop (per the request -- "우하단부터 sweep, 정확도 0.8 이하면 더 낮추지 않음"):
collapsed configs generate to the 32k token cap and the eager attention backend is slow, so we
start at the bottom-right (BF16,BF16 -- the most accurate, fastest cell) and walk toward lower
nmp. Once a cell scores <= ACC_FLOOR (0.8), every not-yet-run cell with lower-or-equal nmp on
BOTH axes is pruned (monotonicity: it can only be worse), so we never launch the slow runs.

Resumable: a cell whose result already exists on disk (either a prior sweep run or an existing
canonical outputs/inference/ dir with the same config hash) is read back instantly, no relaunch.

Usage (inside the quantized-reasoning-models conda env; pick free GPUs -- nmp>=6 weight_cache
needs ~2x48GB, so TP=2):

    CUDA_VISIBLE_DEVICES=0,2 python sweep_ozaki_nmp_grid.py            # run the adaptive sweep
    CUDA_VISIBLE_DEVICES=0,2 python sweep_ozaki_nmp_grid.py --dry-run  # print the plan only
    python sweep_ozaki_nmp_grid.py --print-table                      # re-render from disk

Outputs: outputs/sweep_nmp_grid/results.json (machine-readable) + results.md (the rendered
table), refreshed after every cell. Per-cell stdout/stderr -> logs/sweep_nmp_grid/<cell>.log.
"""
import os
import sys
import glob
import json
import time
import hashlib
import argparse
import subprocess

# --- base config (mirrors configs/ozaki1_fp_per_op_gsm8k.yaml, dataset swapped to MATH-500) ---
MODEL = "./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B"
DATASET = "MATH-500"
RSLT_TYPE = "ozaki1_fp"
GEMM_BITS = 4
BYTE_SPLIT_STYLE = "all_signed_no_clamp"
K = 32
WEIGHT_CACHE = True
LINEAR_NAMES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]  # vLLM-fused linear names

# Grid axis. valid w=4 nmps are 1/3/4/6/9/10/15/16; the table uses these seven + BF16 (native).
BF16 = "BF16"
NMP_VALUES = [1, 3, 4, 6, 9, 10, 15]
AXIS = NMP_VALUES + [BF16]            # least -> most accurate (== rank order)
ACC_FLOOR = 0.8                       # cell <= this => stop lowering nmp past it (prune worse cells)

OUT_DIR = "outputs/sweep_nmp_grid"
LOG_DIR = "logs/sweep_nmp_grid"
INFERENCE_DIR = "outputs/inference"   # where inference_vllm.py writes its canonical hash dirs

# Known-good reference hashes (config -> dir suffix) used as a startup self-check: if the
# canonical hashing below ever drifts from inference_vllm.py::_make_run_tag, this trips loudly
# instead of silently recomputing every cached cell. {hash: (placement, base_nmp, overrides)}
_SELFCHECK = {
    "f2eb474f": ("linear_only", 1, None),                                  # (L=1, A=BF16)=0.912
    "e5b4c52c": ("linear_only", 3, None),                                  # (L=3, A=BF16)=0.936
    "214e6d44": ("full", 6, None),                                         # (L=6, A=6)  =0.918
    "c80fba47": ("full", 6, {n: 1 for n in LINEAR_NAMES}),                 # (L=1, A=6)  =0.426
}


def rank(v):
    """Accuracy rank: higher == more accurate (BF16 highest, nmp=1 lowest)."""
    return AXIS.index(v)


def cell_to_run(L, A):
    """Map a (linear_nmp, attn_nmp) cell to a run spec: placement / base nmp / linear overrides."""
    if L == BF16 and A == BF16:
        return dict(placement="off", nmp=None, overrides=None)
    if A == BF16:                                  # linear emulated, attention native
        return dict(placement="linear_only", nmp=L, overrides=None)
    if L == BF16:                                  # attention emulated, linear native
        return dict(placement="attn_only", nmp=A, overrides=None)
    if L == A:                                     # both emulated at the same nmp
        return dict(placement="full", nmp=L, overrides=None)
    # both emulated, different nmp: base nmp drives attention; per-op overrides pin the linears.
    return dict(placement="full", nmp=A, overrides={n: L for n in LINEAR_NAMES})


def canonical_args(spec):
    """Full args dict (every inference_vllm.py argparse field) for a run spec, so we can compute
    the SAME output-dir hash inference_vllm.py would and reuse any existing matching run.
    Mirrors inference_vllm.py defaults; verified bit-exact against existing runs (see _SELFCHECK).
    Only the non-excluded keys actually affect the hash, but we fill the full set to be safe."""
    placement = spec["placement"]
    a = {
        # --- excluded from the hash (perf / run-control / derived) but kept for completeness ---
        "config": None, "output_dir": None, "output_path": None,
        "model_name": MODEL.rstrip("/").split("/")[-1], "tensor_parallel_size": 2,
        "overwrite": False, "debug": False, "dataset": DATASET,
        "gpu_memory_utilization": 0.9, "max_num_batched_tokens": None, "max_num_seqs": None,
        # --- part of the hash ---
        "model": MODEL, "dtype": "bfloat16", "seed": 42,
        "temperature": 0.6, "top_p": 0.95, "max_new_tokens": 32768, "max_model_length": 32768,
        "max_samples": None, "methods": None, "load_responses_from_json_file": None,
        "ozaki_arch": "Qwen2OzakiForCausalLM",
        "ozaki_placement": (None if placement == "off" else placement),
        "rslt_type": RSLT_TYPE, "k": K, "weight_cache": WEIGHT_CACHE,
        "nmp": (6 if spec["nmp"] is None else spec["nmp"]),  # 6 = inference_vllm default (off: unused)
        "nmp_overrides": spec["overrides"],
        "gemm_bits": GEMM_BITS, "byte_split_style": BYTE_SPLIT_STYLE,
        # ozaki-2 only (ignored for ozaki1_fp; excluded from this scheme's hash)
        "s": None, "s_overrides": None, "scale_method": "new_compressed",
        "shift_bits": 7, "M_frac_bits": 8, "combine_fp64": False,
    }
    return a


def run_tag(a):
    """Replicates inference_vllm.py::_make_run_tag exactly (legible parts + 8-char md5)."""
    if a["ozaki_placement"] is None:
        keys = ["model", "dtype", "seed"]
    elif a["rslt_type"] in ("ozaki1", "ozaki1_fp"):
        keys = ["model", "ozaki_placement", "rslt_type", "nmp", "k", "weight_cache", "dtype", "seed"]
    else:
        keys = ["model", "ozaki_placement", "rslt_type", "s", "k", "scale_method",
                "shift_bits", "M_frac_bits", "weight_cache", "combine_fp64", "dtype", "seed"]
    parts = []
    for k in keys:
        v = a[k]
        if isinstance(v, bool):
            v = int(v)
        elif "/" in str(v):
            v = str(v).rstrip("/").split("/")[-1]
        parts.append(f"{k}={v}")
    exclude = {"config", "output_dir", "output_path", "model_name", "tensor_parallel_size",
               "overwrite", "debug", "dataset", "gpu_memory_utilization",
               "max_num_batched_tokens", "max_num_seqs"}
    ozaki1_only = {"nmp", "nmp_overrides", "gemm_bits", "byte_split_style"}
    ozaki2_only = {"s", "scale_method", "shift_bits", "M_frac_bits", "combine_fp64", "s_overrides"}
    if a["ozaki_placement"] is None:
        exclude |= {"ozaki_placement", "rslt_type", "k", "weight_cache", "ozaki_arch"} | ozaki1_only | ozaki2_only
    elif a["rslt_type"] in ("ozaki1", "ozaki1_fp"):
        exclude |= ozaki2_only
    else:
        exclude |= ozaki1_only
    cfg = {k: v for k, v in a.items() if k not in exclude}
    h = hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return "__".join(parts) + f"__{h}"


def canonical_jsonl(spec):
    """The exact path current inference_vllm.py writes for this spec."""
    return os.path.join(INFERENCE_DIR, run_tag(canonical_args(spec)), f"{DATASET}.jsonl")


def result_jsonl(spec):
    """Best existing result path for a spec, else the canonical path (where a fresh run lands).
    The native bf16 baseline (BF16,BF16) predates inference_vllm.py's `methods` arg, so its dir
    hash differs from today's; for `off` we fall back to any matching native-baseline dir so the
    known 0.948 is reused instead of recomputed."""
    primary = canonical_jsonl(spec)
    if os.path.exists(primary):
        return primary
    if spec["placement"] == "off":
        name = MODEL.rstrip("/").split("/")[-1]
        pat = os.path.join(INFERENCE_DIR, f"model={name}__dtype=bfloat16__seed=42__*", f"{DATASET}.jsonl")
        for j in sorted(glob.glob(pat)):
            aj = os.path.join(os.path.dirname(j), f"{DATASET}.args.json")
            try:
                if json.load(open(aj)).get("ozaki_placement") is None:
                    return j
            except (OSError, json.JSONDecodeError):
                continue
    return primary


def read_accuracy(path):
    """Mean MATH-500 extractive_match over the saved records, or None if absent/empty."""
    if not os.path.exists(path):
        return None
    try:
        recs = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return None
    if not recs:
        return None
    return sum(r["metrics"].get("extractive_match", 0.0) for r in recs) / len(recs)


def build_cmd(spec):
    """inference_vllm.py CLI for a run spec."""
    cmd = [sys.executable, "inference_vllm.py", "--model", MODEL, "--dataset", DATASET]
    if spec["placement"] == "off":
        cmd += ["--ozaki_placement", "off"]
        return cmd
    cmd += ["--ozaki_placement", spec["placement"], "--rslt_type", RSLT_TYPE,
            "--gemm_bits", str(GEMM_BITS), "--byte_split_style", BYTE_SPLIT_STYLE, "--k", str(K)]
    if WEIGHT_CACHE:
        cmd += ["--weight_cache"]
    cmd += ["--nmp", str(spec["nmp"])]
    if spec["overrides"]:
        cmd += ["--nmp_overrides", ",".join(f"{n}={v}" for n, v in spec["overrides"].items())]
    return cmd


def self_check():
    """Fail loudly if our canonical hashing has drifted from inference_vllm.py."""
    bad = []
    for want, (placement, nmp, ov) in _SELFCHECK.items():
        spec = dict(placement=placement, nmp=nmp, overrides=ov)
        got = run_tag(canonical_args(spec)).split("__")[-1]
        if got != want:
            bad.append(f"  {placement} nmp={nmp} ov={ov}: expected {want}, got {got}")
    if bad:
        raise SystemExit("Canonical-hash self-check FAILED (inference_vllm.py changed?):\n"
                         + "\n".join(bad))


# --------------------------------------------------------------------------------------------
# table rendering
# --------------------------------------------------------------------------------------------
def fmt_cell(rec):
    if rec is None:
        return "  ?  "
    st = rec["status"]
    if st in ("done", "reused") and rec.get("acc") is not None:
        return f"{rec['acc']:.3f}"
    if st == "pruned":
        return "prune"
    if st == "failed":
        return "FAIL "
    return "  ?  "


def render_table(results):
    """results: {(L,A): rec}. Rows = linear nmp, cols = attn nmp, both in AXIS order."""
    hdr = "Lin\\Attn | " + " ".join(f"{str(a):>5}" for a in AXIS)
    lines = [hdr, "-" * len(hdr)]
    for L in AXIS:
        row = " ".join(f"{fmt_cell(results.get((L, A))):>5}" for A in AXIS)
        lines.append(f"{str(L):>8} | {row}")
    return "\n".join(lines)


def write_outputs(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    serial = {f"L={L}|A={A}": rec for (L, A), rec in results.items()}
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(serial, f, indent=2, default=str)
    table = render_table(results)
    with open(os.path.join(OUT_DIR, "results.md"), "w") as f:
        f.write(f"# Ozaki1_fp (w=4) nmp sweep — {DATASET} extractive_match\n\n")
        f.write("Rows = linear-layer nmp, cols = attention nmp, BF16 = native. "
                f"`prune` = skipped (dominated by a <= {ACC_FLOOR} cell).\n\n```\n")
        f.write(table + "\n```\n")
    return table


# --------------------------------------------------------------------------------------------
# main sweep
# --------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned order / commands / cache status; launch nothing.")
    ap.add_argument("--print-table", action="store_true",
                    help="Re-read every cell from disk, render the table, and exit.")
    ap.add_argument("--acc-floor", type=float, default=ACC_FLOOR,
                    help=f"Collapse threshold; cells <= this prune lower-nmp cells (default {ACC_FLOOR}).")
    ap.add_argument("--timeout-hours", type=float, default=None,
                    help="Optional per-cell wall-clock cap (a degenerate full/attn run can take days).")
    args = ap.parse_args()
    floor = args.acc_floor

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    self_check()

    # Cells, bottom-right first: highest combined rank (most accurate / fastest) leads, so every
    # strict dominator of a cell is evaluated before it -> the prune below is always well-informed.
    cells = sorted(((L, A) for L in AXIS for A in AXIS),
                   key=lambda c: (rank(c[0]) + rank(c[1]), rank(c[0]), rank(c[1])), reverse=True)

    # Seed results from disk (resume / reuse existing canonical runs).
    results = {}
    for L, A in cells:
        path = result_jsonl(cell_to_run(L, A))
        acc = read_accuracy(path)
        if acc is not None:
            results[(L, A)] = {"status": "reused", "acc": acc, "path": path}

    if args.print_table:
        print(render_table(results))
        return

    failed = [c for c, r in results.items() if r.get("acc") is not None and r["acc"] <= floor]

    def dominated(L, A):
        # pruned iff some failed cell has >= nmp on BOTH axes (so this cell can only be worse)
        for (Lf, Af) in failed:
            if rank(L) <= rank(Lf) and rank(A) <= rank(Af):
                return (Lf, Af)
        return None

    print(f"Sweep: {DATASET}, ozaki1_fp w={GEMM_BITS}, acc_floor={floor}. "
          f"{len(results)} cell(s) already on disk.\n")

    os.makedirs(LOG_DIR, exist_ok=True)
    for idx, (L, A) in enumerate(cells, 1):
        spec = cell_to_run(L, A)
        tag = f"L={L},A={A}"

        if (L, A) in results and results[(L, A)].get("acc") is not None:
            r = results[(L, A)]
            print(f"[{idx:2}/{len(cells)}] {tag:<14} reuse  acc={r['acc']:.3f}")
            continue

        dom = dominated(L, A)
        if dom is not None:
            results[(L, A)] = {"status": "pruned", "dominated_by": f"L={dom[0]},A={dom[1]}"}
            print(f"[{idx:2}/{len(cells)}] {tag:<14} prune  (<= {floor} at L={dom[0]},A={dom[1]})")
            write_outputs(results)
            continue

        cmd = build_cmd(spec)
        logf = os.path.join(LOG_DIR, f"L{L}_A{A}.log")
        if args.dry_run:
            print(f"[{idx:2}/{len(cells)}] {tag:<14} RUN    {' '.join(cmd)}")
            continue

        print(f"[{idx:2}/{len(cells)}] {tag:<14} run    {' '.join(cmd[2:])}\n"
              f"                    log -> {logf}", flush=True)
        t0 = time.time()
        timeout = args.timeout_hours * 3600 if args.timeout_hours else None
        with open(logf, "w") as lf:
            try:
                subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                               check=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                results[(L, A)] = {"status": "failed", "reason": "timeout", "log": logf}
                print(f"                    TIMEOUT after {args.timeout_hours}h — see {logf}")
                write_outputs(results)
                continue
            except subprocess.CalledProcessError as e:
                results[(L, A)] = {"status": "failed", "reason": f"exit {e.returncode}", "log": logf}
                print(f"                    FAILED (exit {e.returncode}) — see {logf}")
                write_outputs(results)
                continue

        path = result_jsonl(spec)
        acc = read_accuracy(path)
        dt = (time.time() - t0) / 60.0
        if acc is None:
            results[(L, A)] = {"status": "failed", "reason": "no_output", "log": logf}
            print(f"                    no result jsonl found — see {logf}")
        else:
            results[(L, A)] = {"status": "done", "acc": acc,
                               "path": path, "log": logf, "minutes": round(dt, 1)}
            print(f"                    acc={acc:.3f}  ({dt:.1f} min)")
            if acc <= floor:
                failed.append((L, A))
                print(f"                    <= {floor}: pruning lower-nmp cells from here.")
        write_outputs(results)

    print("\n" + render_table(results))
    if not args.dry_run:
        print(f"\nWrote {OUT_DIR}/results.json and {OUT_DIR}/results.md")


if __name__ == "__main__":
    main()
