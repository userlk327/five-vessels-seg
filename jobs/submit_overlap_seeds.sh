#!/bin/bash
# Overlapping-tile inference (src.sam_overlap) for tiled SAM seeds 1 and 2, so the sensitivity variant
# has the same 3 training seeds as the main tiled SAM (user request after the 3-seed Phase 6 update).
# Inference only: each seed's own sam_lora_tiled best.pt, nothing is trained.
#
#   job 1 (GPU): val for seeds 1 and 2 -> each seed's own val-selected threshold
#   job 2 (GPU, only if job 1 succeeded): the test split ONCE per seed, scored at that threshold
#                (src.sam_overlap refuses to overwrite an existing test archive)
#   optional (WHOLE=1): whole-image SAM seeds 1 and 2, frozen config unchanged, trained in parallel with
#                TEST_AFTER=1 (train -> val -> test once; 6 h limit, seed 0 took 2.9 h; resumes if requeued)
#   last (CPU, after all of the above succeeded): src.analysis sam and src.report --final-test
#
# PARTITION=gpu: run the GPU jobs on the shared gpu partition (still --constraint=gpu_l40s, from the
# sbatch files) instead of ltarunl, when every L40S on c0903 is held by other users' jobs that an
# ltarunl job would preempt.
#
# Splitting val and test into two jobs means a failed val step can never consume a seed's single
# test prediction (gpu.sbatch exits non-zero if any part fails, so afterok blocks job 2).
set -euo pipefail
WHOLE="${WHOLE:-0}"
S=$(env | grep -oE '^SLURM_[A-Z_]+' | sed 's/^/-u /' || true)
W=(--workers 8)
P=(); [ -n "${PARTITION:-}" ] && P=(--partition="$PARTITION")
v=$(env $S sbatch --parsable "${P[@]}" --job-name=fives-overlap-val --time=00:30:00 jobs/gpu.sbatch \
    src.sam_overlap --seed 1 --split val "${W[@]}" :: src.sam_overlap --seed 2 --split val "${W[@]}")
t=$(env $S sbatch --parsable "${P[@]}" --dependency=afterok:$v --job-name=fives-overlap-test --time=00:30:00 jobs/gpu.sbatch \
    src.sam_overlap --seed 1 --split test --final-test "${W[@]}" :: \
    src.sam_overlap --seed 2 --split test --final-test "${W[@]}")
deps="$t"
echo "overlap val  seeds 1,2  job $v"
echo "overlap test seeds 1,2  job $t  (after $v succeeded)"
if [ "$WHOLE" = 1 ]; then
    for s in 1 2; do
        j=$(env $S TEST_AFTER=1 sbatch --parsable "${P[@]}" --time=06:00:00 --job-name=fives-sam_lora_whole-s$s jobs/train.sbatch \
            configs/sam_lora_whole.yaml $s)
        deps="$deps:$j"
        echo "sam_lora_whole seed $s   job $j  (train -> val -> test once)"
    done
fi
r=$(env $S sbatch --parsable --dependency=afterok:$deps --job-name=fives-report --time=00:30:00 \
    jobs/cpu.sbatch src.analysis sam :: src.report --final-test)
echo "report                  job $r  (after $deps)"
