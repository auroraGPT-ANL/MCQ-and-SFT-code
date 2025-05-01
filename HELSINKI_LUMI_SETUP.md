## README: Setting Up Conda on LUMI

To use Conda properly on Lumi:

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

Make sure to edit the *YOUR_PATH* portion to match your
path on the NVIDIA cluster, where $HOME expands to */users/your_username* (use your username).

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

(or log out and log back in.)

---

### 3. Create your Conda environment (first time only)

From your project directory:

```bash
cotainr build mcq.sif --system=lumi-g --conda-env=environment.yml
```

**(This will take a while— 10–15 minutes.)**

---

### 4. Activate the environment

Once created (or on future logins), you'll be running the codes as follows:

```bash
singularity exec mcq.sif python -m [script] [options]
```
Or, if submitting a batch job:
```bash
srun singularity exec mcq.sif python -m [script] [options]
```

**(CeC is still testing this)**

You are now ready to work!

