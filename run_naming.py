"""Canonical output-dir naming for inference runs.

Single source of truth for the run-tag hash, shared by ``inference_vllm.py`` (which
WRITES the dirs) and any tool that needs to LOCATE them (e.g. ``sweep_ozaki_nmp_grid.py``).
Keeping this in its own module — with no heavyweight deps (no torch/vllm/transformers) —
means locating a run's dir never forces those imports, and the two sides can no longer
drift out of a hand-copied hasher.
"""
import json
import hashlib


def make_run_tag(args):
    """Output-dir name for a run: legible ``key=value`` parts + an 8-char md5 of the full
    config. Mirrors emulation/llm/utils.py::make_result_filename so runs are self-describing
    and any two distinct configs land in distinct dirs. The dataset is the file name
    (<dataset>.jsonl) INSIDE the dir, not part of the tag, so one config dir collects every
    dataset run with that config. Perf-only knobs (gpu mem util, batch caps) and run-control
    flags are excluded from the hash since they don't change the results.

    Accepts either an ``argparse.Namespace`` (inference_vllm.py) or a plain ``dict`` (tools
    that reconstruct a run's args), so both callers hash identically.
    """
    a = args if isinstance(args, dict) else vars(args)
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
    # Hash the config that actually affects THIS run. Always drop run-control / derived / perf
    # fields; additionally drop Ozaki params that don't apply (all of them when ozaki is off,
    # the other scheme's params when on) so numerically-identical runs share one dir.
    # NOTE: the scheduler knobs below are excluded to keep run dirs stable, but they are NOT
    # numerically neutral -- they change the decode batch, hence linear-GEMM shapes (bf16 reduction
    # order) and the flash split-KV count. Two runs differing only in these land in the SAME dir, so
    # give a matched-scheduler baseline an explicit --output_dir instead of relying on the hash.
    exclude = {"config", "output_dir", "output_path", "model_name", "tensor_parallel_size",
               "overwrite", "debug", "dataset", "gpu_memory_utilization",
               "max_num_batched_tokens", "max_num_seqs", "enable_chunked_prefill"}
    # pack_dtype only exists from 2026-08; excluding it at its default keeps every pre-existing run
    # dir's hash intact, while an fp16 run still lands in its own dir (it can change results at w=5).
    if a.get("pack_dtype", "bf16") == "bf16":
        exclude |= {"pack_dtype"}
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
