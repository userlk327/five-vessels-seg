#!/bin/bash
# Build the single pinned Python environment used by every job.
# DO NOT RUN without approval (Rule 8). Run once, from the project root, on the
# login node (download/install only, no computation):
#     bash jobs/setup_env.sh
# It creates a conda env at $ENV_PREFIX (default $WORK/fives-vessel-seg/env),
# installs requirements.txt with pip, and writes the full resolved package list
# to requirements.lock.txt (commit that file).
set -eo pipefail   # no -u: Lmod's module function reads unset variables
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FIVES_NO_ACTIVATE=1 source jobs/env.sh
module purge
module load anaconda
eval "$(conda shell.bash hook)"

if [ -d "$ENV_PREFIX" ]; then
    echo "ERROR: $ENV_PREFIX already exists. Remove it first if you want a clean rebuild." >&2
    exit 1
fi

conda create -y -p "$ENV_PREFIX" python=3.11.18 pip
conda activate "$ENV_PREFIX"
python -m pip install -r requirements.txt
python -m pip freeze --all > requirements.lock.txt

python - <<'EOF'
import torch, torchvision, cv2, skimage, scipy, numpy, pandas, segment_anything
print("python      ", __import__("sys").version.split()[0])
print("torch       ", torch.__version__, "| CUDA", torch.version.cuda)
print("compiled for", torch._C._cuda_getArchFlags(), "(V100 = sm_70; L40S = sm_89, runs sm_86 code)")
print("torchvision ", torchvision.__version__)
print("numpy", numpy.__version__, "| scipy", scipy.__version__, "| skimage", skimage.__version__,
      "| cv2", cv2.__version__, "| pandas", pandas.__version__)
EOF
echo "Env ready at $ENV_PREFIX. Next: sbatch jobs/env_check.sbatch (GPU check)."
