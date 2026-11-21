#!/bin/bash
# Tiled SAM, seeds 1 and 2: the frozen config (configs/sam_lora_tiled.yaml) unchanged (user request after Phase 6).
#   Each training job (jobs/train.sbatch, TEST_AFTER=1): train -> predict + score val (that seed's own val
#   threshold) -> predict the test split ONCE and score it at that threshold, all on the job's own GPU, so no
#   extra GPU job is needed (none can preempt anyone).
#   PARALLEL=1 (default): both seeds at once (only when 2 L40S are free); PARALLEL=0: seed 2 after seed 1.
#   Then a CPU job (after both succeeded): src.analysis sam (val table) and src.report --final-test (all
#   Phase 6 tables and figures, tiled SAM now as 3-seed means).
set -euo pipefail
PARALLEL="${PARALLEL:-1}"
S=$(env | grep -oE '^SLURM_[A-Z_]+' | sed 's/^/-u /' || true)
sb() { env $S TEST_AFTER=1 sbatch --parsable "$@"; }
t1=$(sb --job-name=fives-sam_lora_tiled-s1 jobs/train.sbatch configs/sam_lora_tiled.yaml 1)
dep=""; [ "$PARALLEL" = 1 ] || dep="--dependency=afterany:$t1"
t2=$(sb $dep --job-name=fives-sam_lora_tiled-s2 jobs/train.sbatch configs/sam_lora_tiled.yaml 2)
r=$(env $S sbatch --parsable --dependency=afterok:$t1:$t2 --job-name=fives-report --time=00:30:00 \
    jobs/cpu.sbatch src.analysis sam :: src.report --final-test)
echo "sam_lora_tiled seed 1  job $t1"
echo "sam_lora_tiled seed 2  job $t2  ($([ "$PARALLEL" = 1 ] && echo parallel || echo "after $t1"))"
echo "report                 job $r  (after both)"
