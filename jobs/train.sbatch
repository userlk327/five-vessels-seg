#!/bin/bash
#SBATCH --job-name=fives-train
#SBATCH --partition=ltarunl          # group-owned partition: no preemption, 7-day max
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_l40s        # same GPU type for every model -> comparable time/memory numbers
#SBATCH --cpus-per-task=8            # DataLoader workers; 4 such jobs fit on the 4-GPU L40S node
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append           # one log file across requeues
#SBATCH --requeue
#SBATCH --signal=B:USR1@600          # send USR1 to this script 10 min before the time limit
#
# Usage (from the project root):
#     sbatch jobs/train.sbatch configs/unet.yaml 0
#
# Resume: src/train.py always continues from outputs/<model>/seed<seed>/last.pt if
# it exists, so a job that hit its time limit can simply be resubmitted with the
# same arguments. In addition, this script requeues itself near the time limit
# (up to MAX_RESTARTS times), so long runs continue without manual resubmission.
# A run that already finished (early stop or max epochs) exits immediately.
#
# After a finished run (exit code 0), the same job also
#   1. predicts the VAL split from best.pt  (outputs/<run>/seed<seed>/val/probs.npz)
#   2. scores it at 2048                     (results/eval/<run>/seed<seed>/val_*)
# The test split is never predicted here (Phase 6 only).
# Optional: PROBE_RES="512 1024" sbatch ... also runs the GPU memory / speed probe.

CONFIG="${1:?usage: sbatch jobs/train.sbatch <config.yaml> <seed>}"
SEED="${2:?usage: sbatch jobs/train.sbatch <config.yaml> <seed>}"
MAX_RESTARTS="${MAX_RESTARTS:-10}"

source jobs/common.sh

python -m src.train --config "$CONFIG" --seed "$SEED" &
PID=$!

on_time_limit() {
    echo "== $(date -Is) time limit near: stopping training; last.pt (last finished epoch) is kept"
    kill -TERM "$PID" 2>/dev/null
    wait "$PID"
    if [ "${SLURM_RESTART_COUNT:-0}" -lt "$MAX_RESTARTS" ]; then
        echo "== requeueing job $SLURM_JOB_ID"
        scontrol requeue "$SLURM_JOB_ID"
    else
        echo "== restart limit ($MAX_RESTARTS) reached; resubmit by hand to continue"
    fi
    exit 0
}
trap on_time_limit USR1

wait "$PID"
RC=$?
echo "== $(date -Is) training process exited with code $RC"
[ "$RC" -eq 0 ] || exit $RC

echo "== $(date -Is) predicting the val split from best.pt"
python -m src.predict --config "$CONFIG" --seed "$SEED" --split val || exit $?
echo "== $(date -Is) scoring the val split"
python -m src.evaluate score --config "$CONFIG" --seed "$SEED" --split val --workers "${SLURM_CPUS_PER_TASK:-8}" || exit $?
if [ -n "${PROBE_RES:-}" ]; then
    echo "== $(date -Is) GPU memory / speed probe at: $PROBE_RES"
    python -m src.train --config "$CONFIG" --seed "$SEED" --probe $PROBE_RES || echo "== probe failed (training results unaffected)"
fi
echo "== $(date -Is) done"
