#!/bin/bash
# Phase 4 training schedule (approved): run from the project root on Swan.
#   1. seed 0 of Attention, Multi-Scale and MS-Attention U-Net, one after another
#      (clean training times; U-Net seed 0 already exists)
#   2. then seeds 1 and 2 of all four CNNs in CONCURRENCY parallel chains
#      (2 by default; 4 only if nobody else needs the ltarunl GPUs)
# Each job trains, then predicts and scores the val split (jobs/train.sbatch).
# afterany: a failed run does not block the rest of the schedule.
set -euo pipefail
CONCURRENCY="${CONCURRENCY:-2}"
S=$(env | grep -oE '^SLURM_[A-Z_]+' | sed 's/^/-u /' || true)
sb() { env $S sbatch --parsable "$@"; }
prev=""
for m in attention_unet multiscale_unet ms_attention_unet; do
    prev=$(sb ${prev:+--dependency=afterany:$prev} --job-name="fives-$m-s0" jobs/train.sbatch "configs/$m.yaml" 0)
    echo "seed 0  $m  job $prev"
done
runs=()
for seed in 1 2; do for m in unet attention_unet multiscale_unet ms_attention_unet; do runs+=("$m $seed"); done; done
declare -a tail
for ((c = 0; c < CONCURRENCY; c++)); do tail[$c]=$prev; done
i=0
for r in "${runs[@]}"; do
    set -- $r
    c=$((i % CONCURRENCY))
    tail[$c]=$(sb --dependency=afterany:${tail[$c]} --job-name="fives-$1-s$2" jobs/train.sbatch "configs/$1.yaml" "$2")
    echo "seed $2  $1  job ${tail[$c]}  (chain $c)"
    i=$((i + 1))
done
