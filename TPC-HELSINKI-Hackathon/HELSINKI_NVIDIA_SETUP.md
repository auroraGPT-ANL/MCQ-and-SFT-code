## README: Setting Up Conda on Curiosity Cluster

To use Conda properly on the Curiosity cluster login node:

### 1. Add these lines to your `~/.bashrc`

At the very bottom of `~/.bashrc`, add:

```bash
# Load shared conda module
module load conda

# Source conda setup manually
source /cm/shared/apps/conda/etc/profile.d/conda.sh

# set PYTHONPATH for MCQ pipeline at MCQ-and-SFT-code
export PYTHONPATH="$HOME/MCQ-and-SFT-code:$HOME/YOUR_PATH/MCQ-and-SFT-code/src${PYTHONPATH:+:$PYTHONPATH}"
```

> The `module load conda` makes the Conda executable available.  
> The `source` command enables `conda activate`.
> The $PYTHONPATH is needed for the mcq pipleine modules.

Make sure to edit the *YOUR_PATH* portion to match your
path on the NVIDIA cluster, where $HOME expands to */home/tpc-user* (use your username).

```bssh
# If you cloned MCQ-and-SFT-code in your home directory
export PYTHONPATH="$HOME/MCQ-and-SFT-code:$HOME/MCQ-and-SFT-code/src${PYTHONPATH:+:$PYTHONPATH}"
```

```bssh
# If you cloned MCQ-and-SFT-code cloned in, e.g., ~/MyCode 
export PYTHONPATH="$HOME/MCQ-and-SFT-code:$HOME/MyCode/MCQ-and-SFT-code/src${PYTHONPATH:+:$PYTHONPATH}"
```

---

### 2. After editing `.bashrc`, reload it

```bash
source ~/.bashrc
```

(or just log out and log back in.)


> NOTE: For some ssh clients it seems the prompts get messed up, while they do not for other clients.
This does not affect the conda env or the code - it's just annoying.  `¯\_(ツ)_/¯`

---

### 3. Create your Conda environment (first time only)

From your project directory:

```bash
conda env create -f environment.yml
```

**(This will take a while — easily 30–60 minutes.)**

---

### 4. Activate the environment

Once created (or on future logins):

```bash
conda activate augpt_env
```

You are now ready to work!

---

## Full Steps Summary for New Users

| Step | Command or Action |
|:-----|:------------------|
| 1. Edit `~/.bashrc` | Add `module load conda` and `source /cm/shared/apps/conda/etc/profile.d/conda.sh` |
| 2. Reload bashrc | `source ~/.bashrc` |
| 3. Create environment | `conda env create -f environment.yml` |
| 4. Activate environment | `conda activate augpt_env` |

---

## Notes

- You only run `conda env create` **once**.
- After that, just **activate** the environment every time you log in (`conda activate augpt_env`).
- If you change `environment.yml` later, update your environment with:
  
  ```bash
  conda env update -f environment.yml --prune
  ```


