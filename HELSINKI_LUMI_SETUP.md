## README: Setting Up Conda on LUMI

On Lumi everything is executed in containers, so you'll create a container with your conda env and then
exec that container.

### 1. Add these lines to your `~/.bashrc`

At the bottom of your `~/.bashrc`, add:

```bash
# Load shared conda module
module load LUMI
module load lumi-container-wrapper

# set PYTHONPATH for MCQ pipeline at MCQ-and-SFT-code
export PYTHONPATH="$HOME/MCQ-and-SFT-code:$HOME/YOUR_PATH/MCQ-and-SFT-code/src${PYTHONPATH:+:$PYTHONPATH}"
```

Make sure to edit the *YOUR_PATH* portion to match your
path on the NVIDIA cluster, where $HOME expands to */users/your_username* (use your username).

If you cloned MCQ-and-SFT-code in your home directory:
```bash
export PYTHONPATH="$HOME/MCQ-and-SFT-code:$HOME/MCQ-and-SFT-code/src${PYTHONPATH:+:$PYTHONPATH}"
```

If you cloned MCQ-and-SFT-code cloned in, e.g., ~/MyCode:
```bash
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

> This can take a while— building our env took 10–15 minutes. You can also skip
this step and try using the environment noted in */scratch* described below.

---

### 4. Using the environment

Once created (or on future logins), you'll be running the codes as follows:

To run on the login node:
```bash
singularity exec mcq.sif python -m [script] [options]
```
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

From the interactive session you established with srun, you can now
run the scripts with:
```bash
singularity exec --rocm mcq.sif python -m [script] [options]
```

## LUMI Resource Reservation (from Aleksi)

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
