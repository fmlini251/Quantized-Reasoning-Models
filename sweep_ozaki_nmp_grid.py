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

Targeting one nmp first (--priority-nmp N pulls N's row+column to the front, rest of the grid
follows in the usual order). --only-priority stops after them, which is what lets two sweeps share
one results file: `linear` takes the row, `attn` takes the column minus the shared corner cell, so
the two cell sets are disjoint and every write re-harvests the other's finished cells from disk.

    CUDA_VISIBLE_DEVICES=0,1 python sweep_ozaki_nmp_grid.py --gemm-bits 5 --priority-nmp 3
    CUDA_VISIBLE_DEVICES=0,1 python sweep_ozaki_nmp_grid.py --gemm-bits 5 --priority-nmp 3 \
        --priority-axis linear --only-priority        # the L=3 row
    CUDA_VISIBLE_DEVICES=2,3 python sweep_ozaki_nmp_grid.py --gemm-bits 5 --priority-nmp 3 \
        --priority-axis attn   --only-priority        # the A=3 column (minus L=3,A=3)

Outputs: outputs/sweep_nmp_grid/results.json (machine-readable) + results.md (the rendered
table), refreshed after every cell. Per-cell stdout/stderr -> logs/sweep_nmp_grid/<cell>.log.
"""
import os
import sys
import glob
import json
import time
import argparse
import subprocess

from run_naming import make_run_tag  # shared with inference_vllm.py; single source of truth

# --- base config (mirrors configs/ozaki1_fp_per_op_gsm8k.yaml, dataset swapped to MATH-500) ---
MODEL = "./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B"
DATASET = "MATH-500"
RSLT_TYPE = "ozaki1_fp"
GEMM_BITS = 4
BYTE_SPLIT_STYLE = "all_signed_no_clamp"
K = 32
WEIGHT_CACHE = True
USE_FLASH = True                     # run attention (attn_only/full cells) through the Triton
#                                      flash_ozaki kernel (--ozaki_flash) instead of eager batched_gemm:
#                                      faster + avoids the eager attention runtime-OOM. Only affects
#                                      cells whose attention is emulated; linear_only/off cells are
#                                      unchanged (still reused from disk). Distinct run-hash from eager,
#                                      so flash cells run FRESH (never reuse an eager result).
LINEAR_NAMES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]  # vLLM-fused linear names
# Names pinned to the LINEAR nmp (L) in an L!=A `full` cell. lm_head is a linear op (its logits GEMM is
# now Ozaki-fied) so it must follow L, not the base --nmp (=A, the attention nmp). Its module name is
# "lm_head", which does NOT re.search-match any of qkv/o/gate_up/down -> without this it would fall back
# to the base nmp = A (wrong). Keep LINEAR_NAMES (4) for the _SELFCHECK legacy-hash refs below.
OVERRIDE_NAMES = LINEAR_NAMES + ["lm_head"]

# Grid axis. valid w=4 nmps are 1/3/4/6/9/10/15/16; the table uses these seven + BF16 (native).
BF16 = "BF16"
NMP_VALUES = [1, 3, 4, 6, 9, 10, 15]
AXIS = NMP_VALUES + [BF16]            # least -> most accurate (== rank order)
ACC_FLOOR = 0.8                       # cell <= this => stop lowering nmp past it (prune worse cells)
ALL_CELLS = []                        # the full 64-cell grid; set by main(), read by merge_disk()

OUT_DIR = "outputs/sweep_nmp_grid"
LOG_DIR = "logs/sweep_nmp_grid"
PACK_DTYPE = "bf16"                   # super-digit packing dtype for the flash kernel (bf16|fp16)


def _res_names():
    """Result filenames. w=4 + bf16 keeps the historical names so the existing grid is never
    clobbered; any other (w, pack_dtype) gets its own pair of files."""
    base = "results_flash" if USE_FLASH else "results"
    if GEMM_BITS != 4:
        base += f"_w{GEMM_BITS}"
    if PACK_DTYPE != "bf16":
        base += f"_{PACK_DTYPE}"
    return base + ".json", base + ".md"
INFERENCE_DIR = "outputs/inference"   # where inference_vllm.py writes its canonical hash dirs

# Known-good reference hashes (config -> dir suffix) used as a startup self-check. The hash
# ALGORITHM is now imported from inference_vllm.py (run_naming.make_run_tag) so it cannot silently
# drift; what this guards is canonical_args below (a stray/removed hashed field). These four are
# LEGACY dirs — written by the older inference_vllm.py that carried a `--methods` arg, which left a
# `methods=None` entry in the hashed config — so self_check() reproduces them via the same legacy
# shim result_jsonl() uses to reuse them. {hash: (placement, base_nmp, overrides)}
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
    # both emulated, different nmp: base nmp drives attention; per-op overrides pin the linears
    # (incl. lm_head) to L. attn_weights/attn_output don't match these patterns -> attention stays A.
    return dict(placement="full", nmp=A, overrides={n: L for n in OVERRIDE_NAMES})


def canonical_args(spec):
    """Full args dict (every current inference_vllm.py argparse field) for a run spec, so
    make_run_tag() computes the SAME output-dir hash a freshly-launched inference_vllm.py run
    would, and a just-finished run is picked up instead of re-launched. Mirrors inference_vllm.py
    defaults; the hashed keys must stay in lock-step with its argparse (self_check guards this).
    NOTE: no `methods` key — current inference_vllm.py has no `--methods` arg, so adding one here
    would compute a hash no fresh run ever writes (that was the silent-loss bug). Legacy dirs that
    DO carry methods=None are reached via result_jsonl()'s legacy shim, not from here."""
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
        "max_samples": None, "load_responses_from_json_file": None,
        "ozaki_arch": "Qwen2OzakiForCausalLM",
        "ozaki_placement": (None if placement == "off" else placement),
        "rslt_type": RSLT_TYPE, "k": K, "weight_cache": WEIGHT_CACHE,
        "nmp": (6 if spec["nmp"] is None else spec["nmp"]),  # 6 = inference_vllm default (off: unused)
        "nmp_overrides": spec["overrides"],
        "gemm_bits": GEMM_BITS, "byte_split_style": BYTE_SPLIT_STYLE,
        "pack_dtype": PACK_DTYPE,
        # ozaki-2 only (ignored for ozaki1_fp; excluded from this scheme's hash)
        "s": None, "s_overrides": None, "scale_method": "new_compressed",
        "shift_bits": 7, "M_frac_bits": 8, "combine_fp64": False,
    }
    if USE_FLASH and placement in ("attn_only", "full"):
        a["ozaki_flash"] = True          # flash-emulated attention -> distinct run-hash from eager
    return a


def canonical_jsonl(spec):
    """The exact path CURRENT inference_vllm.py writes for this spec (shared make_run_tag), i.e.
    where a freshly-launched run will land and where a just-finished run must be looked up."""
    return os.path.join(INFERENCE_DIR, make_run_tag(canonical_args(spec)), f"{DATASET}.jsonl")


def legacy_jsonl(spec):
    """Path from the OLDER inference_vllm.py that still had a `--methods` arg: it left a
    `methods=None` entry in the hashed config, so its dirs differ from today's. Every result
    currently on disk was written that way, so we resolve it here to reuse completed cells
    instead of relaunching multi-hour runs."""
    a = dict(canonical_args(spec))
    a["methods"] = None
    return os.path.join(INFERENCE_DIR, make_run_tag(a), f"{DATASET}.jsonl")


def _cell_fields(spec):
    """The saved-args fields that DEFINE a sweep cell. Everything else inference_vllm.py writes
    (perf flags, and newly-added hashed args -- `methods`, `kv_cache_prefill`, `weight_cache_prefill`,
    ...) must NOT gate the match: gating on inference_vllm's FULL arg set via a hand-copied replica
    (canonical_args) is exactly what silently broke harvest every time it gained a flag (a completed
    run's hash then no longer matched -> 'no_output'). Values as written into <dataset>.args.json."""
    p = spec["placement"]
    if p == "off":
        return {"ozaki_placement": None, "dtype": "bfloat16", "seed": 42}
    f = {"ozaki_placement": p, "rslt_type": RSLT_TYPE, "k": K, "weight_cache": WEIGHT_CACHE,
         "nmp": spec["nmp"], "nmp_overrides": spec["overrides"], "gemm_bits": GEMM_BITS,
         "byte_split_style": BYTE_SPLIT_STYLE, "dtype": "bfloat16", "seed": 42}
    if PACK_DTYPE != "bf16":
        f["pack_dtype"] = PACK_DTYPE      # fp16 packing cells must not match a bf16 run
    if USE_FLASH and p in ("attn_only", "full"):
        f["ozaki_flash"] = True           # flash cells must match a flash run, never the eager one
    return f


def _field_eq(saved, want):
    """Saved-args value vs wanted cell-field value; bool-tolerant (missing/None counts as False)."""
    if isinstance(want, bool):
        return bool(saved) == want
    return saved == want                  # None==None, int==int, dict==dict (order-independent)


_DISK = None                          # cached [(jsonl_path, saved_args)] of every run dir on disk


def disk_scan(refresh=False):
    """Every finished run dir as (jsonl path, saved args), read once and cached: a full re-harvest
    touches this list 64 times (once per cell), and re-globbing/re-parsing per cell made that
    O(cells x dirs) file reads."""
    global _DISK
    if _DISK is None or refresh:
        name = MODEL.rstrip("/").split("/")[-1]
        scan = []
        for j in sorted(glob.glob(os.path.join(INFERENCE_DIR, f"model={name}__*", f"{DATASET}.jsonl"))):
            aj = os.path.join(os.path.dirname(j), f"{DATASET}.args.json")
            try:
                scan.append((j, json.load(open(aj))))
            except (OSError, json.JSONDecodeError):
                continue
        _DISK = scan
    return _DISK


def result_jsonl(spec, refresh=False):
    """Existing result jsonl for a spec, located by scanning the model's run dirs and confirming the
    CELL-DEFINING fields in each candidate's <dataset>.args.json -- robust to inference_vllm.py adding
    new hashed-but-non-result args (the repeated cause of 'no_output' on completed cells). Falls back
    to the canonical path (where a fresh run will land) when nothing on disk matches yet. Pass
    refresh=True right after a run finishes -- its dir is newer than the cached scan."""
    want = _cell_fields(spec)
    for j, saved in disk_scan(refresh):
        if all(_field_eq(saved.get(k), v) for k, v in want.items()):
            return j
    return canonical_jsonl(spec)


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
    if USE_FLASH and spec["placement"] in ("attn_only", "full"):
        cmd += ["--ozaki_flash"]
    if PACK_DTYPE != "bf16":
        cmd += ["--pack_dtype", PACK_DTYPE]
    return cmd


def self_check():
    """Fail loudly if canonical_args has drifted. The hash algorithm is imported from
    inference_vllm.py (run_naming.make_run_tag) so it can't silently diverge; the remaining risk
    is canonical_args gaining/losing a hashed field. The four references are LEGACY dirs, so we
    reproduce them through the same `methods=None` shim legacy_jsonl() uses -- if canonical_args
    changes shape this trips, instead of silently missing every cached cell."""
    global GEMM_BITS, PACK_DTYPE
    _w, _pd = GEMM_BITS, PACK_DTYPE
    GEMM_BITS, PACK_DTYPE = 4, "bf16"     # the four reference dirs are w=4/bf16 legacy runs
    bad = []
    for want, (placement, nmp, ov) in _SELFCHECK.items():
        spec = dict(placement=placement, nmp=nmp, overrides=ov)
        a = dict(canonical_args(spec))
        a.pop("ozaki_flash", None)       # refs are eager legacy; check the flash-independent base hash
        a["methods"] = None
        a.pop("pack_dtype", None)        # legacy dirs predate pack_dtype
        got = make_run_tag(a).split("__")[-1]
        if got != want:
            bad.append(f"  {placement} nmp={nmp} ov={ov}: expected {want}, got {got}")
    GEMM_BITS, PACK_DTYPE = _w, _pd
    if bad:
        raise SystemExit("Canonical-hash self-check FAILED (canonical_args changed?):\n"
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


def merge_disk(results):
    """Fold every cell that now has a result on disk but no accuracy in `results` back in. Called
    before each write so a sweep restricted to part of the grid (--only-priority) still renders the
    whole table, and so two concurrent sweeps sharing one results file pick up each other's finished
    cells instead of clobbering them."""
    disk_scan(refresh=True)
    for c in ALL_CELLS:
        if results.get(c, {}).get("acc") is not None:
            continue
        path = result_jsonl(cell_to_run(*c))
        acc = read_accuracy(path)
        if acc is not None:
            results[c] = {"status": "reused", "acc": acc, "path": path}
    return results


def _write_atomic(path, text):
    """Write via temp + rename: concurrent sweeps then never read a half-written file."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def write_outputs(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    merge_disk(results)
    serial = {f"L={L}|A={A}": rec for (L, A), rec in results.items()}
    RES_JSON, RES_MD = _res_names()
    _write_atomic(os.path.join(OUT_DIR, RES_JSON), json.dumps(serial, indent=2, default=str))
    table = render_table(results)
    _write_atomic(os.path.join(OUT_DIR, RES_MD),
                  f"# Ozaki1_fp (w={GEMM_BITS}) nmp sweep — {DATASET} extractive_match\n\n"
                  "Rows = linear-layer nmp, cols = attention nmp, BF16 = native. "
                  f"`prune` = skipped (dominated by a <= {ACC_FLOOR} cell).\n\n```\n"
                  + table + "\n```\n")
    return table


# --------------------------------------------------------------------------------------------
# main sweep
# --------------------------------------------------------------------------------------------
def main():
    global GEMM_BITS, PACK_DTYPE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned order / commands / cache status; launch nothing.")
    ap.add_argument("--print-table", action="store_true",
                    help="Re-read every cell from disk, render the table, and exit.")
    ap.add_argument("--acc-floor", type=float, default=ACC_FLOOR,
                    help=f"Collapse threshold; cells <= this prune lower-nmp cells (default {ACC_FLOOR}).")
    ap.add_argument("--gemm-bits", type=int, default=4, choices=[2, 3, 4, 5, 6, 7, 8],
                    help="Ozaki-1 GEMM unit width w for BOTH axes (default 4). "
                         "int_bits = w*nD-1 sets the accuracy ladder, so a different w is a "
                         "different experiment and gets its own results_*.json/md.")
    ap.add_argument("--pack-dtype", type=str, default="bf16", choices=["bf16", "fp16"],
                    help="Super-digit packing dtype of the flash ATTENTION kernel (linear layers go "
                         "through production, which is always bf16-packed). Only changes anything "
                         "when w does not divide 8 (w=5: g 1->2).")
    ap.add_argument("--timeout-hours", type=float, default=None,
                    help="Optional per-cell wall-clock cap (a degenerate full/attn run can take days).")
    ap.add_argument("--priority-nmp", type=str, default=None,
                    help="Comma-separated nmp value(s) whose row/column jump to the front of the "
                         "queue (e.g. 3). The rest of the grid follows in the normal order.")
    ap.add_argument("--priority-axis", choices=["both", "linear", "attn"], default="both",
                    help="Which cells --priority-nmp selects: the row AND column (both, default), "
                         "the LINEAR row only, or the ATTENTION column only. `attn` also drops the "
                         "cells already in the priority row, so `linear` and `attn` split the "
                         "row+column into two disjoint halves that can run side by side.")
    ap.add_argument("--only-priority", action="store_true",
                    help="Run ONLY the --priority-nmp cells and exit (the table still renders the "
                         "whole grid from disk).")
    args = ap.parse_args()
    GEMM_BITS, PACK_DTYPE = args.gemm_bits, args.pack_dtype
    floor = args.acc_floor

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    self_check()

    # Cells, bottom-right first: highest combined rank (most accurate / fastest) leads, so every
    # strict dominator of a cell is evaluated before it -> the prune below is always well-informed.
    global ALL_CELLS
    ALL_CELLS = sorted(((L, A) for L in AXIS for A in AXIS),
                       key=lambda c: (rank(c[0]) + rank(c[1]), rank(c[0]), rank(c[1])), reverse=True)

    # Seed results from disk (resume / reuse existing canonical runs). Always over the FULL grid, so
    # the rendered table keeps every cell even when this process only runs part of it.
    results = {}
    for L, A in ALL_CELLS:
        path = result_jsonl(cell_to_run(L, A))
        acc = read_accuracy(path)
        if acc is not None:
            results[(L, A)] = {"status": "reused", "acc": acc, "path": path}

    if args.print_table:
        print(render_table(results))
        return

    # Optional re-ordering: pull one nmp's row/column to the front. ALL_CELLS is already
    # rank-descending, so a STABLE partition keeps bottom-right-first order within each group. The
    # prune below stays sound either way (it only fires on an actually-observed <= floor cell), it
    # just has fewer dominators in hand when a priority cell runs out of rank order.
    cells = list(ALL_CELLS)
    if args.priority_nmp:
        prio = [int(v) for v in args.priority_nmp.split(",") if v.strip()]
        bad = [v for v in prio if v not in NMP_VALUES]
        if bad:
            raise SystemExit(f"--priority-nmp: {bad} not a grid nmp {NMP_VALUES}")

        def in_prio(c):
            L, A = c
            if args.priority_axis == "linear":
                return L in prio
            if args.priority_axis == "attn":
                return A in prio and L not in prio   # disjoint from the `linear` half
            return L in prio or A in prio

        cells.sort(key=lambda c: 0 if in_prio(c) else 1)
        if args.only_priority:
            cells = [c for c in cells if in_prio(c)]
        print(f"Priority: nmp={prio} ({args.priority_axis}) first"
              + (" — ONLY these cells" if args.only_priority else "")
              + f" -> {sum(1 for c in cells if in_prio(c))} cell(s).")

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

        path = result_jsonl(spec, refresh=True)   # this run's dir is newer than the cached scan
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
        res_json, res_md = _res_names()      # local: write_outputs() computes its own copy
        print(f"\nWrote {OUT_DIR}/{res_json} and {OUT_DIR}/{res_md}"
              + ("   [FLASH attention sweep]" if USE_FLASH else ""))


if __name__ == "__main__":
    main()
