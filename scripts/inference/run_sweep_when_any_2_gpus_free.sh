#!/bin/bash
# Wait until ANY N GPUs are idle, then launch the ozaki nmp-grid sweep on exactly those GPUs.
# Unlike run_sweep_when_gpus_free.sh (which waits on a FIXED device list), this takes whichever
# GPUs free up first -- use it when you don't care which pair you get.
#
# Start detached so it survives the launching shell / agent session:
#   mkdir -p logs/sweep_nmp_grid_w5
#   nohup setsid scripts/inference/run_sweep_when_any_2_gpus_free.sh 2 --gemm-bits 5 \
#       >> logs/sweep_nmp_grid_w5/master.log 2>&1 < /dev/null &
#
# $1        = how many GPUs to wait for (default 2; the sweep runs TP = #visible GPUs)
# $2..      = passed straight through to sweep_ozaki_nmp_grid.py
# FREE_MIB  = a GPU counts as free below this used-memory (default 2000; idle reports ~5-120 MiB)
set -uo pipefail

NGPU=${1:-2}; shift || true
FREE_MIB=${FREE_MIB:-2000}
POLL_SEC=${POLL_SEC:-120}
ENVPY=${ENVPY:-/home/howonlee/.conda/envs/quantized-reasoning-models/bin/python}

cd "$(dirname "$0")/../.."

echo "[$(date '+%F %T')] waiter armed: need ${NGPU} GPU(s) with used<${FREE_MIB}MiB (poll ${POLL_SEC}s)"
echo "[$(date '+%F %T')] sweep args: $*"
while true; do
    free=()
    while IFS=, read -r idx used; do
        idx=$(echo "$idx" | tr -d ' '); used=$(echo "$used" | tr -d ' ')
        [[ "$used" =~ ^[0-9]+$ ]] && [ "$used" -lt "$FREE_MIB" ] && free+=("$idx")
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
    if [ "${#free[@]}" -ge "$NGPU" ]; then
        devs=$(IFS=,; echo "${free[*]:0:$NGPU}")
        echo "[$(date '+%F %T')] free GPUs [${free[*]}] -> launching on CUDA_VISIBLE_DEVICES=$devs"
        break
    fi
    sleep "$POLL_SEC"
done

CUDA_VISIBLE_DEVICES="$devs" VLLM_WORKER_MULTIPROC_METHOD=spawn \
    "$ENVPY" -u sweep_ozaki_nmp_grid.py "$@"
rc=$?
echo "[$(date '+%F %T')] sweep exited rc=$rc"
exit $rc
