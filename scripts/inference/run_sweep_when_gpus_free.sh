#!/bin/bash
# Wait until the given GPUs are idle, then launch the ozaki nmp-grid sweep on them.
# Meant to be started detached (nohup setsid ...) so it survives the launching shell:
#
#   mkdir -p logs/sweep_nmp_grid
#   nohup setsid scripts/inference/run_sweep_when_gpus_free.sh "1,3" \
#       >> logs/sweep_nmp_grid/master.log 2>&1 &
#
# A GPU counts as "free" once its used memory drops below ${FREE_MIB:-2000} MiB
# (idle GPUs report ~120 MiB here; the jobs we wait on hold ~48 GB).
set -uo pipefail

devices=${1:-1,3}                 # CUDA_VISIBLE_DEVICES for the sweep + the GPUs to wait on
FREE_MIB=${FREE_MIB:-2000}
POLL_SEC=${POLL_SEC:-120}
ENVPY=${ENVPY:-/home/howonlee/.conda/envs/quantized-reasoning-models/bin/python}

cd "$(dirname "$0")/../.."

used_mib() {  # echo used MiB for GPU index $1, or a big number if the query fails
    local v
    v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null)
    [[ "$v" =~ ^[0-9]+$ ]] && echo "$v" || echo 999999
}

IFS=',' read -ra GPUS <<< "$devices"
echo "[$(date '+%F %T')] waiter armed; will run on GPUs [$devices] once each has used<${FREE_MIB}MiB (poll ${POLL_SEC}s)"
while true; do
    all_free=1; report=""
    for g in "${GPUS[@]}"; do
        u=$(used_mib "$g"); report+="gpu$g=${u}MiB "
        [ "$u" -ge "$FREE_MIB" ] && all_free=0
    done
    [ "$all_free" -eq 1 ] && break
    sleep "$POLL_SEC"
done
echo "[$(date '+%F %T')] GPUs free ($report) -> launching sweep on CUDA_VISIBLE_DEVICES=$devices"

CUDA_VISIBLE_DEVICES="$devices" VLLM_WORKER_MULTIPROC_METHOD=spawn \
    "$ENVPY" sweep_ozaki_nmp_grid.py
echo "[$(date '+%F %T')] sweep process exited code=$?"
