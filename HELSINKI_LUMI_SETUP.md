## README: Setting Up Conda on LUMI

On Lumi everything is executed in containers. We have identified
a pre-built container on Lumi and we use it in conjuntion with
our conda environment.

### 1. Update your ~/.bashrc


```bash
# ~/.bashrc - for Lumi ROCm + Singularity + Conda + Hugging Face

# Load aliases if present
test -s ~/.alias && . ~/.alias || true

# Load CrayEnv only if not inside a container
if [[ -z "$SINGULARITY_NAME" ]]; then
    module load CrayEnv
    module load cotainr
fi

# Enable conda
export PATH="$HOME/miniconda/bin:$PATH"
eval "$(conda shell.bash hook)"

# Use project scratch space
export MYSCRATCH=/scratch/project_465001984

# Set Conda & pip caches to scratch
export CONDA_ENVS_PATH=$MYSCRATCH/conda_envs
export CONDA_PKGS_DIRS=$MYSCRATCH/conda_pkgs
export PIP_CACHE_DIR=$MYSCRATCH/pip_cache

# Set Hugging Face cache to scratch
export HF_HOME=$MYSCRATCH/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export HF_MODULES_CACHE=$HF_HOME/modules
export HF_METRICS_CACHE=$HF_HOME/metrics

```
---

### 2. After editing `.bashrc`, reload it

```bash
source ~/.bashrc
```

(or log out and log back in.)

---

### 3. Move into the root repo directory (MCQ-and-SFT-code

If you cloned the repo from your ~/ ($HOME) directory:
```bash
cd MCQ-and-SFT-code
```

### 4. Create your Conda environment (first time only)

```bash
conda env create -f rocm.yml
```

> This can take 5-10 minutes. 

### 5. Create the launch script

Create the file `launch_augpt_env.sh` in the MCQ-and-SFT-code repo root
directory with the following contents:

```bash
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

```

Make it executable:
```bash
chmod +x launch_augpt_env.sh
```

---

### 6. Request compute resources and establish the runtime environment (shell)

E.g., for a single node shell session on Lumi:

```bash
srun -A project_465001984 -p standard-g      --nodes=1 --gres=gpu:1 --cpus-per-task=4 --mem=64G      --time=02:00:00      --pty bash -i
./launch_augpt_env.sh
```
### 7. Using the environment

Once created (or on future logins), you'll be running the codes as shown
in the root README.md of the MCQ-and-SFT-code repo directory.

---

## other potentially useful incantations and notes

To run on a Lumi compute node (with one-shot resource allocation)::
```bash
srun -A project_465001984 -p standard-g \
     --nodes=1 --gres=gpu:1 --cpus-per-task=4 --mem=64G \
     --time=02:00:00 \
     singularity exec --rocm mcq.sif \
       python -m [script] [options]
```

Get node resources and fire up an interactive session on them:
```bash
srun -A project_465001984 -p standard-g \
     --nodes=1 --gres=gpu:1 --cpus-per-task=4 --mem=64G \
     --time=02:00:00 \
     --pty bash -i
```

## LUMI Resource Reservation for the Hackathon (May 2025)(from Aleksi)

We have an advance reservation of 8 full nodes (4x AMD MI250X GPU / 8 GCD’s) in
**small-g** partition.

It is available from 2025-05-06T09:00:00 to 2025-05-08T17:00:00.

Use Slurm parameter: `--reservation=TPC`

Using the reservation bypasses queues (as long as there is capacity left within the
reservation). Submitting jobs without using the reservation is also possible.

## Shared environment in /scratch

The hackathon project scratch area is `/scratch/project_465001984`.  
There I have created a `MCQ` directory and in it placed a singularity image
(mcq.sif) with the conda env for running our codes..

It is 7GB so you mignt want to use this one
in case you want to save disk space in your home dir (20GB limit).
Or you may just want to cp to your $HOME rather than building one (which takes 15-20 min).

To run our scripts using the .sif file there, once you have cloned the repo in your home
directory and set up .bashrc as noted above::

```bash
singularity exec --rocm /scratch/project_465001984/MCQ/mcq.sif python -m [SCRIPT] [OPTIONS/ARGS]
```

For example:
```bash
singularity exec --rocm /scratch/project_465001984/MCQ/mcq.sif python -m mcq_workflow.run_workflow -v
```

> --rocm tells singularity to use the AMD/ROCm libraries wherease if Lumi had NVIDIA GPU's you would use --nv
