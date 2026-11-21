#!/bin/bash
#SBATCH --job-name=fives-eval
#SBATCH --partition=batch            # metrics at 2048 (skeletons, distance transforms) are CPU work
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#
# Threshold selection on val, then per-image metrics at 2048 -> results/ CSVs.
#     sbatch jobs/evaluate.sbatch configs/unet.yaml 0 val
# Arguments after <seed> are passed to src/evaluate.py (interface defined in Phase 2).

CONFIG="${1:?usage: sbatch jobs/evaluate.sbatch <config.yaml> <seed> [args...]}"
SEED="${2:?usage: sbatch jobs/evaluate.sbatch <config.yaml> <seed> [args...]}"
shift 2

source jobs/common.sh
python -m src.evaluate --config "$CONFIG" --seed "$SEED" "$@"
