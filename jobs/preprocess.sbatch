#!/bin/bash
#SBATCH --job-name=fives-prep
#SBATCH --partition=batch            # CPU-only work (PNG extraction, CLAHE, resizing, mask caches)
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#
# Phase 1/2 data preparation. Arguments are passed straight to src/data.py, e.g.
#     sbatch jobs/preprocess.sbatch extract --config configs/base.yaml
# (subcommands are defined in Phase 1).

source jobs/common.sh
python -m src.data "$@"
