#!/usr/bin/env python
"""
MCQ workflow driver – now using ProcessPoolExecutor instead of shell “&”.

CLI stays the same:
    python -m mcq_workflow.run_workflow -p 8 -v
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def run_sync(cmd: str) -> int:
    """Run a shell command synchronously, echoing it, return exit code."""
    print(f"Running: {cmd}")
    completed = subprocess.run(cmd, shell=True)
    if completed.returncode != 0:
        print(f"\n❌  Command failed: {cmd}\n   Exit code: {completed.returncode}")
    return completed.returncode


def run_step4_job(job: tuple[str, str, int, str]) -> int:
    """Function executed in workers for step 4."""
    model, input_file, p_value, v_flag = job
    cmd = (
        f"python -m mcq_workflow.generate_answers "
        f"-i {input_file} -m {model} -p {p_value} {v_flag}"
    )
    return run_sync(cmd)


def main() -> None:
    start_time = time.time()

    # project root = two levels up from this file
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    # secrets.yml sanity check
    if not (project_root / "secrets.yml").exists():
        print("Please create secrets.yml. If using Argo models, populate with:\n"
              "argo:\n    username: YOUR_ARGO_USERNAME")
        sys.exit(1)

    # Add src/ to PYTHONPATH
    os.environ["PYTHONPATH"] = f"{project_root/'src'}:{os.environ.get('PYTHONPATH','')}"

    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=8, help="Threads for generation/scoring")
    parser.add_argument("-v", action="store_true", help="Verbose mode")
    parser.add_argument("-n", type=int, help="Select n random MCQs")
    parser.add_argument("-s", "--step", type=int, default=1, help="Step number to start from (1–5)")
    parser.add_argument("-a", "--answers", type=int, help="Number of answer choices per MCQ")
    args = parser.parse_args()

    p_value = args.p
    v_flag = "-v" if args.v else ""
    n_value = args.n
    start_step = args.step
    num_answers = args.answers

    if start_step > 1:
        print(f"⏩ Skipping to step {start_step}\n")

    # ---------- Step 1
    if start_step <= 1:
        print("Step 1: Convert PDF to JSON")
        if run_sync("python -m common.simple_parse"):
            sys.exit(1)

    # ---------- Step 2
    if start_step <= 2:
        print("Step 2: Generate MCQs")
        ans_opt = f"-a {num_answers}" if num_answers is not None else ""
        if run_sync(f"python -m mcq_workflow.generate_mcqs -p {p_value} {ans_opt} {v_flag}"):
            sys.exit(1)

    # ---------- Step 3
    if start_step <= 3:
        print("Step 3: Combine JSON files")
        if run_sync("python -m common.combine_json_files -o MCQ-combined.json"):
            sys.exit(1)

        input_file = "MCQ-combined.json"
        if n_value:
            print(f"Selecting {n_value} MCQs at random…")
            if run_sync(f"python -m common.select_mcqs_at_random -i MCQ-combined.json "
                        f"-o MCQ-subset.json -n {n_value}"):
                sys.exit(1)
            input_file = "MCQ-subset.json"
    else:
        input_file = "MCQ-subset.json" if n_value else "MCQ-combined.json"

    # ---------- Step 4   (parallel)
    if start_step <= 4:
        print("Step 4: Generate answers (all models)")
        result = subprocess.run(
            f"python -m common.list_models -p {p_value}",
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )
        models = result.stdout.strip().splitlines()

        jobs = [(m, input_file, p_value, v_flag) for m in models]
        with ProcessPoolExecutor(max_workers=p_value) as pool:
            futures = {pool.submit(run_step4_job, job): job[0] for job in jobs}
            for fut in as_completed(futures):
                model = futures[fut]
                code = fut.result()
                if code != 0:
                    print(f"⛔  Aborting – model {model} failed")
                    sys.exit(code)

    # ---------- Step 5
    if start_step <= 5:
        print("Step 5: Retrieve answers for all models")
        if glob.glob('_RESULTS/answers*'):
            if run_sync(f"python -m mcq_workflow.retrieve_results {v_flag}"):
                sys.exit(1)

    # ---------- Timing
    elapsed = int(time.time() - start_time)
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Total elapsed time: {hrs:02}:{mins:02}:{secs:02} (hh:mm:ss)")


if __name__ == "__main__":
    main()

