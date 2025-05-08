#!/usr/bin/env python

""" run_workflow
Simplified Python wrapper for the MCQ workflow.
All subprocesses write directly to your terminal, with optional step-skipping.
"""

import os
import sys
import subprocess
import time
import argparse


def run(cmd, background=False):
    """Run a shell command, optionally in background."""
    print(f"Running: {cmd}")
    if background:
        return subprocess.Popen(cmd, shell=True)
    else:
        subprocess.run(cmd, shell=True, check=True)


def main():
    start_time = time.time()

    # Determine project root (two levels up)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # Ensure secrets.yml exists
    secrets_file = os.path.join(project_root, "secrets.yml")
    if not os.path.isfile(secrets_file):
        print("Please create secrets.yml. If using Argo models, populate with:")
        print("argo:\n    username: YOUR_ARGO_USERNAME")
        sys.exit(1)

    # Add src/ to PYTHONPATH
    src_dir = os.path.join(project_root, "src")
    os.environ["PYTHONPATH"] = f"{src_dir}:{os.environ.get('PYTHONPATH', '')}"

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=8,
                        help="Threads for generation/scoring")
    parser.add_argument("-v", action="store_true",
                        help="Verbose mode (passes -v to child scripts)")
    parser.add_argument("-n", type=int,
                        help="Select n random MCQs")
    parser.add_argument("-s", "--step", type=int, default=1,
                        help="Step number to start from (1–5)")
    parser.add_argument("-a", "--answers", type=int, default=7,
                        help="Number of answers to generate")
    args = parser.parse_args()

    p_value = args.p
    v_flag = "-v" if args.v else ""
    n_value = args.n
    start_step = args.step
    num_answers = args.answers

    # If starting beyond step 1, print skip banner
    if start_step > 1:
        print(f"⏩ Skipping to step {start_step}\n")

    # Prepare default input_file if skipping step 3
    if start_step > 3:
        input_file = "MCQ-subset.json" if n_value else "MCQ-combined.json"

    # Step 1: Convert PDF to JSON
    if start_step <= 1:
        print("Step 1: Convert PDF to JSON")
        run("python -m common.simple_parse")

    # Step 2: Generate MCQs
    if start_step <= 2:
        print("Step 2: Generate MCQs")
        run(f"python -m mcq_workflow.generate_mcqs -p {p_value} -a {num_answers} {v_flag}")

    # Step 3: Combine JSON files
    if start_step <= 3:
        print("Step 3: Combine JSON files")
        run("python -m common.combine_json_files -o MCQ-combined.json")

        # Optional: select subset if requested
        if n_value:
            print(f"Selecting {n_value} MCQs at random...")
            run(
                f"python -m mcq_workflow.select_mcqs_at_random "
                f"-i MCQ-combined.jsonl -o MCQ-subset.jsonl -n {n_value}"
            )
            input_file = "MCQ-subset.json"
        else:
            input_file = "MCQ-combined.json"

    # Step 4: Generate answers for all models
    if start_step <= 4:
        print("Step 4: Generate answers (all models)")
        # Determine input file if not set (in case start_step == 4)
        if 'input_file' not in locals():
            input_file = "MCQ-subset.json" if n_value else "MCQ-combined.json"
        processes = []
        result = subprocess.run(
            f"python -m common.list_models -p {p_value}",
            shell=True, capture_output=True, text=True, check=True
        )
        MODELS = result.stdout.strip().splitlines()
        for model in MODELS:
            cmd = (
                f"python -m mcq_workflow.generate_answers "
                f"-i {input_file} -m {model} -p {p_value} {v_flag}"
            )
            processes.append(run(cmd, background=True))
        for p in processes:
            p.wait()
            if p.returncode != 0:
                sys.exit(p.returncode)

    # Step 5: Score answers between all models
    if start_step <= 5:
        print("Step 5: Score answers between all models")
        # Determine input file if not set (in case start_step >3)
        if 'input_file' not in locals():
            input_file = "MCQ-subset.json" if n_value else "MCQ-combined.json"
        processes = []
        result = subprocess.run(
            f"python -m common.list_models -p {p_value}",
            shell=True, capture_output=True, text=True, check=True
        )
        MODELS = result.stdout.strip().splitlines()
        for i, a in enumerate(MODELS, start=1):
            for j, b in enumerate(MODELS, start=1):
                if i != j:
                    print(f"Scoring Model {i} answers using Model {j}")
                    cmd = (
                        f"python -m mcq_workflow.score_answers "
                        f"-a {a} -b {b} -p {p_value} {v_flag}"
                    )
                    processes.append(run(cmd, background=True))
        for p in processes:
            p.wait()
            if p.returncode != 0:
                sys.exit(p.returncode)

    # Final timing
    elapsed = int(time.time() - start_time)
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Total elapsed time: {hrs:02}:{mins:02}:{secs:02} (hh:mm:ss)")


if __name__ == "__main__":
    main()

