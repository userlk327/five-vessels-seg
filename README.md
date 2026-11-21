# fives-vessel-seg

Retinal vessel segmentation on FIVES (Jin et al., *Scientific Data* 2022): a U-Net
baseline, a 2×2 ablation (attention × multi-scale) and SAM ViT-B, all under one
protocol, with a focus on thin vessels. The plan is in `CLAUDE.md`; every protocol
decision is in `docs/protocol.md`.

## Setup on Swan (once per person)

```bash
git clone <repo-url> && cd fives-vessel-seg
FIVES_DIR=/path/to/FIVES bash jobs/setup_links.sh   # $WORK folders + symlinks (outputs/, logs/, data/cache, data/FIVES)
bash jobs/setup_env.sh                              # pinned conda env under $WORK
sbatch jobs/env_check.sbatch                        # GPU check
```

## Running (always from the project root)

```bash
sbatch jobs/train.sbatch configs/unet.yaml 0        # resumes automatically if resubmitted
sbatch jobs/predict.sbatch configs/unet.yaml 0 val
pytest                                              # unit tests (CPU, fine on the login node)
```

Interactive GPU session:
`srun --partition=ltarunl --gres=gpu:1 --constraint=gpu_l40s --cpus-per-task=8 --mem=60G --time=02:00:00 --pty bash`,
then `source jobs/env.sh`.

## Layout

| Path | Contents | Where it lives |
|---|---|---|
| `configs/` | `base.yaml` (data/eval), `train_shared.yaml` (training protocol), one YAML per model | git |
| `data/splits/splits.csv` | the one train/val/test split used by everyone | git |
| `src/` | data, models, losses, train, predict, metrics, evaluate, analysis | git |
| `jobs/` | SLURM scripts and env setup | git |
| `results/`, `docs/` | CSVs, tables, figures; protocol | git |
| `outputs/`, `logs/`, `data/cache/`, `data/FIVES/` | checkpoints, probability maps, logs, caches, data | `$WORK` (symlinks, git-ignored) |
