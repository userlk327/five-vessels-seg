#!/bin/bash
#SBATCH --job-name=fives-cpu
#SBATCH --partition=batch            # CPU-only work: FOV masks, metrics over the dataset, analyses
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#
# Run one or more module commands in sequence, separated by "::", e.g.
#     sbatch jobs/cpu.sbatch src.fov all :: src.evaluate widths
# Each part is run as: python -m <module> <args...>

source jobs/common.sh
set -e
cmd=()
run() { echo "== $(date -Is) python -m ${cmd[*]}"; python -m "${cmd[@]}"; }
for a in "$@"; do
    if [ "$a" = "::" ]; then run; cmd=(); else cmd+=("$a"); fi
done
[ ${#cmd[@]} -gt 0 ] && run
echo "== $(date -Is) done"
