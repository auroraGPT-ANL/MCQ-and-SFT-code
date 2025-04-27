#!/usr/bin/env python

""" run_mcq_workflow
Simplified Python wrapper for the MCQ workflow.
All subprocesses write directly to your terminal.
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

    # Locate project root (two levels up)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # Ensure secrets.yml exists
    secrets_file = os.path.join(project_root, "secrets.yml")
    if not os.path.isfile(secrets_file):
        print("Please create secrets.yml. If using argo models, populate with:")
        print("argo:\n    username: YOUR_ARGO_USERNAME")
        sys.exit(1)

    # Add src/ to PYTHONPATH
    src_dir = os.path.join(project_root, "src")
    os.environ["PYTHONPATH"] = f"{src_dir}:{os.environ.get('PYTHONPATH', '')}"

    # Parse CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=8,
                        help="Threads for generation/scoring")
    parser.add_argument("-v", action="store_true",
                        help="Verbose mode (passes -v to child scripts)")
    parser.add_argument("-n", type=int,
                        help="Select n random MCQs")
    args = parser.parse_args()

    p_value = args.p
    v_flag = "-v" if args.v else ""
    n_value = args.n

    # Step 0: list models
    result = subprocess.run(
        f"python -m common.list_models -p {p_value}",
        shell=True, capture_output=True, text=True, check=True
    )
    MODELS = result.stdout.strip().splitlines()
    ALIASES = ["Model A", "Model B", "Model C", "Model D"]

    print("Models to be used:")
    for i, model in enumerate(MODELS, start=1):
        alias = ALIASES[i-1] if i-1 < len(ALIASES) else f"Model {i}"
        print(f"  {alias}: {model}")

    print(f"Options: -p {p_value}" + (f"  -v" if args.v else "") +
          (f"  -n {n_value}" if n_value else ""))

    # Step 1: Convert PDF → JSON
    print("Step 1: Convert PDF to JSON")
    run("python -m common.simple_parse")

    # Step 2: Generate MCQs
    print("Step 2: Generate MCQs")
    run(f"python -m mcq_workflow.generate_mcqs -p {p_value} {v_flag}")

    # Step 3: Combine JSON files
    print("Step 3: Combine JSON files")
    run("python -m common.combine_json_files -o MCQ-combined.json")

    # Optional subset
    if n_value:
        print(f"Selecting {n_value} MCQs at random")
        run(
            f"python -m mcq_workflow.select_mcqs_at_random "
            f"-i MCQ-combined.jsonl -o MCQ-subset.jsonl -n {n_value}"
        )
        input_file = "MCQ-subset.json"
    else:
        input_file = "MCQ-combined.json"

    # Step 4: Generate answers in parallel
    print("Step 4: Generate answers (all models)")
    processes = []
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

    # Step 5: Score answers in parallel
    print("Step 5: Score answers between all models")
    processes = []
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

