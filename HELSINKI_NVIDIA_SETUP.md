# README: Setting Up Conda on Curiosity Cluster

To use Conda properly on the Curiosity cluster login node:

## 1. Add these lines to your `~/.bashrc`

At the very bottom of `~/.bashrc`, add:

```bash
# Load shared conda module
module load conda

# Source conda setup manually
source /cm/shared/apps/conda/etc/profile.d/conda.sh
```

> The `module load conda` makes the Conda executable available.  
> The `source` command enables `conda activate`.

---

## 2. After editing `.bashrc`, reload it

```bash
source ~/.bashrc
```

(or just log out and log back in.)

---

## 3. Create your Conda environment (first time only)

From your project directory:

```bash
conda env create -f environment.yml
```

*(This may take a while — even 30–60 minutes.)*

---

## 4. Activate the environment

Once created (or on future logins):

```bash
conda activate augpt_env
```

You are now ready to work!

---

# Full Steps Summary for New Users

| Step | Command or Action |
|:-----|:------------------|
| 1. Edit `~/.bashrc` | Add `module load conda` and `source /cm/shared/apps/conda/etc/profile.d/conda.sh` |
| 2. Reload bashrc | `source ~/.bashrc` |
| 3. Create environment | `conda env create -f environment.yml` |
| 4. Activate environment | `conda activate augpt_env` |

---

# Notes

- You only run `conda env create` **once**.
- After that, **just activate** the environment every time you log in.
- If you change `environment.yml` later, update your environment with:
  
  ```bash
  conda env update -f environment.yml --prune
  ```


