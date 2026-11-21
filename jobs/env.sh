# Shared environment for every job script and interactive session.
# Usage:  source jobs/env.sh            (from the project root)
#
# Everything large lives under $PROJECT_WORK on $WORK (fast, NOT backed up,
# purged after ~6 months without access). Code, configs and results stay in
# the project folder in $HOME (backed up, 20 GiB quota, not for job I/O).

if [ -z "${WORK:-}" ]; then
    echo "ERROR: \$WORK is not set (are you on Swan?)" >&2
    return 1 2>/dev/null || exit 1
fi

export PROJECT_WORK="${PROJECT_WORK:-$WORK/fives-vessel-seg}"
# Conda env location. Default is $WORK, as agreed. HCC suggests $NRDSTOR for large
# envs (not purged); to use it, export ENV_PREFIX=$NRDSTOR/fives-vessel-seg-env before sourcing.
export ENV_PREFIX="${ENV_PREFIX:-$PROJECT_WORK/env}"

# Keep every software cache off $HOME (quota) and on $WORK.
export PIP_CACHE_DIR="$PROJECT_WORK/sw_cache/pip"
export TORCH_HOME="$PROJECT_WORK/sw_cache/torch"
export HF_HOME="$PROJECT_WORK/sw_cache/hf"
export MPLCONFIGDIR="$PROJECT_WORK/sw_cache/matplotlib"
export XDG_CACHE_HOME="$PROJECT_WORK/sw_cache/xdg"
export CONDA_PKGS_DIRS="$PROJECT_WORK/sw_cache/conda_pkgs"
mkdir -p "$PIP_CACHE_DIR" "$TORCH_HOME" "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$CONDA_PKGS_DIRS"

# Ignore packages in ~/.local so only the pinned env is used.
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
# One thread per allocated CPU (login node: 1).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENCV_NUM_THREADS="$OMP_NUM_THREADS"

# Activate the env (skip with FIVES_NO_ACTIVATE=1, e.g. while building it).
if [ "${FIVES_NO_ACTIVATE:-0}" != "1" ]; then
    module purge
    module load anaconda
    module load git          # purge unloads it; run_info.jsonl records the git commit (Rule 7)
    eval "$(conda shell.bash hook)"
    if [ ! -d "$ENV_PREFIX" ]; then
        echo "ERROR: conda env not found at $ENV_PREFIX. Build it with jobs/setup_env.sh first." >&2
        return 1 2>/dev/null || exit 1
    fi
    conda activate "$ENV_PREFIX"
fi
