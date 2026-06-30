#!/bin/bash
# Linear-nmp x Attention-nmp accuracy sweep for ozaki1_fp (w=4) on MATH-500.
# Adaptive: starts bottom-right (BF16,BF16) and stops lowering nmp once a cell scores <= 0.8.
# Reuses any existing outputs/inference/ run with a matching config hash (free resume).
#
# Usage:
#   scripts/inference/sweep_nmp_grid.sh "0,2"            # run the sweep on GPUs 0,2 (TP=2)
#   scripts/inference/sweep_nmp_grid.sh "0,2" --dry-run  # print the plan only
#   scripts/inference/sweep_nmp_grid.sh "" --print-table # just render the table from disk
#
# nmp>=6 with weight_cache needs ~2x48GB, so pass two free GPUs. Run inside the
# quantized-reasoning-models conda env (the vLLM + emulation build lives there).
set -euo pipefail

devices=${1:-0,2}   # CUDA_VISIBLE_DEVICES; inference_vllm.py sets TP = #devices
shift || true       # forward any remaining flags (--dry-run / --print-table / --acc-floor ...)

cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES=${devices} \
    python sweep_ozaki_nmp_grid.py "$@"
