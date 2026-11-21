#!/bin/bash
#SBATCH --job-name=fives-predict
#SBATCH --partition=ltarunl
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_l40s        # same GPU type as training (inference-time numbers)
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#
# Save probability maps from a run's best checkpoint.
#     sbatch jobs/predict.sbatch configs/unet.yaml 0 val
# The test split is predicted only in Phase 6 (Rule 3).

CONFIG="${1:?usage: sbatch jobs/predict.sbatch <config.yaml> <seed> <val|test>}"
SEED="${2:?usage: sbatch jobs/predict.sbatch <config.yaml> <seed> <val|test>}"
SPLIT="${3:?usage: sbatch jobs/predict.sbatch <config.yaml> <seed> <val|test>}"

source jobs/common.sh
python -m src.predict --config "$CONFIG" --seed "$SEED" --split "$SPLIT"
