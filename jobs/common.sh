# Sourced by every .sbatch script: go to the project root, activate the env,
# print what is running where. Submit jobs from the project root, e.g.
#     sbatch jobs/train.sbatch configs/unet.yaml 0
# (--output=logs/... in the headers is relative to the submission folder;
#  logs/ is a symlink to $WORK/fives-vessel-seg/logs, created by setup_links.sh.)

cd "${SLURM_SUBMIT_DIR:-.}"
if [ ! -f src/train.py ] || [ ! -L logs ]; then
    echo "ERROR: submit from the project root, after running jobs/setup_links.sh" >&2
    exit 1
fi
source jobs/env.sh || exit 1

echo "== job ${SLURM_JOB_ID:-none} (${SLURM_JOB_NAME:-}) on $(hostname) at $(date -Is)," \
     "restart ${SLURM_RESTART_COUNT:-0}, partition ${SLURM_JOB_PARTITION:-?}, cpus ${SLURM_CPUS_PER_TASK:-?}"
echo "== git: $(git rev-parse --short HEAD 2>/dev/null || echo 'no commit')" \
     "$(git diff --quiet HEAD 2>/dev/null || echo '(uncommitted changes)')"
echo "== python: $(command -v python)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi
