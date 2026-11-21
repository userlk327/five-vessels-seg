#!/bin/bash
#SBATCH --job-name=fives-gpu
#SBATCH --partition=ltarunl
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_l40s        # same GPU type as all training runs
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#
# Short GPU checks: one or more module commands separated by "::", e.g.
#     sbatch jobs/gpu.sbatch src.sam_train gradcheck --config configs/sam_lora_tiled.yaml :: src.sam_oracle --split val
# Each part runs as: python -m <module> <args...>; a failing part does not stop the others.

source jobs/common.sh
cmd=()
rc=0
run() { echo "== $(date -Is) python -m ${cmd[*]}"; python -m "${cmd[@]}" || { echo "== FAILED: ${cmd[*]}"; rc=1; }; }
for a in "$@"; do
    if [ "$a" = "::" ]; then run; cmd=(); else cmd+=("$a"); fi
done
[ ${#cmd[@]} -gt 0 ] && run
echo "== $(date -Is) done (rc $rc)"
exit $rc
