#!/bin/bash
echo "🔧 Checking node type..."

# ❌ Prevent running on login/UAN nodes
if [[ $(hostname) == uan* ]]; then
  echo "❌ ERROR: This script must be run from a compute node (not from a login/UAN node like $(hostname))."
  echo "💡 Use 'salloc' or 'srun' to allocate a compute node session before running this script."
  exit 1
fi
echo "✅ Running on a compute node: $(hostname)"

# ╭────────────────────────────────────────────────────────────╮
# │ 0. Locate HF token                                         │
# ╰────────────────────────────────────────────────────────────╯
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOKEN_FILE="$SCRIPT_DIR/hf_access_token.txt"
[[ -f "$TOKEN_FILE" ]] || { echo "❌ Token file not found at $TOKEN_FILE"; exit 1; }
export HUGGINGFACE_TOKEN=$(<"$TOKEN_FILE")

# ╭────────────────────────────────────────────────────────────╮
# │ 1. Scratch caches                                          │
# ╰────────────────────────────────────────────────────────────╯
echo "📦 Setting up scratch-based cache paths..."
export MYSCRATCH=/scratch/project_465001984
export CONDA_ENVS_PATH=$MYSCRATCH/conda_envs
export CONDA_PKGS_DIRS=$MYSCRATCH/conda_pkgs
export PIP_CACHE_DIR=$MYSCRATCH/pip_cache
export HF_HOME=$MYSCRATCH/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export HF_MODULES_CACHE=$HF_HOME/modules
export HF_METRICS_CACHE=$HF_HOME/metrics

# ╭────────────────────────────────────────────────────────────╮
# │ 2. Paths                                                   │
# ╰────────────────────────────────────────────────────────────╯
echo "📁 Setting container, conda, and project paths..."
export MYSIF=/appl/local/containers/tested-containers/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35-dockerhash-3cad1babc4b8.sif

MYCONDA="$HOME/miniconda"
MYENV="augpt_env"
# ensure both your repo root *and* its src/ are on PYTHONPATH
MYPY="$HOME/MCQ-and-SFT-code:$HOME/MCQ-and-SFT-code/src:$HOME/Dropbox/MyCode/ALCF/MCQ-and-SFT-code/src"

# ╭────────────────────────────────────────────────────────────╮
# │ 3. Launch container                                        │
# ╰────────────────────────────────────────────────────────────╯
echo "🚀 Launching container and initializing interactive environment..."
singularity exec --rocm \
  -B "$HOME","$MYSCRATCH" \
  --env HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
  "$MYSIF" bash --rcfile <(cat <<'EOF'
# — inside the container shell —

# re-declare so they exist here
MYCONDA="$HOME/miniconda"
MYENV="augpt_env"
MYPY="$HOME/MCQ-and-SFT-code:$HOME/MCQ-and-SFT-code/src:$HOME/Dropbox/MyCode/ALCF/MCQ-and-SFT-code/src"

echo "📡 Sourcing Conda environment..."
source "$MYCONDA/etc/profile.d/conda.sh"
conda activate "$MYENV" && echo "✅ Conda environment '$MYENV' activated."

# pass through all cache locations + token
export CONDA_ENVS_PATH="$CONDA_ENVS_PATH"
export CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS"
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export HF_HOME="$HF_HOME"
export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export HF_MODULES_CACHE="$HF_MODULES_CACHE"
export HF_METRICS_CACHE="$HF_METRICS_CACHE"
export HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN"

# set PYTHONPATH so your package under src/ is always importable
export PYTHONPATH="$MYPY${PYTHONPATH:+:$PYTHONPATH}"
echo "📚 PYTHONPATH set to: $PYTHONPATH"

# final prompt
PS1="($MYENV Singularity) \\u@\\h:\\w\\$ "
EOF
)

