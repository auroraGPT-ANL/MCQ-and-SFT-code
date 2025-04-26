#!/usr/bin/env python

""" run_mcq_workflow
Python version of MCQ workflow, avoiding an overly complex .sh script
(with the peculiarities of bash vs zsh and sourcing vs. executing).
All models are specified in config.yml (at the root of the repo).
Usernames and tokens are kept in secrets.yml (also at root).
"""

import os
import sys
import subprocess
import time
import argparse


def run(cmd, background=False, kill_on_warning=False):
    print(f"Running: {cmd}")
    if background:
        return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    else:
        with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end='')
                if kill_on_warning and ("WARNING:" in line or "warning:" in line):
                    print(f"Detected WARNING, killing workflow: {line.strip()}")
                    proc.terminate()
                    sys.exit(1)
            proc.wait()
            if proc.returncode != 0:
                sys.exit(proc.returncode)


def main():
    start_time = time.time()

    # Determine project root (two levels up)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    secrets_file = os.path.join(project_root, "secrets.yml")
    if not os.path.isfile(secrets_file):
        print("Please create secrets.yml. If using argo models, populate with:")
        print("""argo:
    username: YOUR_ARGO_USERNAME""")
        sys.exit(1)

    # Set PYTHONPATH
    src_dir = os.path.join(project_root, "src")
    os.environ["PYTHONPATH"] = f"{src_dir}:{os.environ.get('PYTHONPATH', '')}"

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=8, help="Threads for generation/scoring")
    parser.add_argument("-v", action="store_true", help="Verbose mode")
    parser.add_argument("-n", type=int, help="Select n random MCQs")
    parser.add_argument("-k", "--kill-on-warning", action="store_true", help="Exit immediately if any WARNING is detected")
    args = parser.parse_args()

    p_value = args.p
    v_flag = "-v" if args.v else ""
    n_value = args.n
    kill_on_warning = args.kill_on_warning

    # List models
    print("Listing models...")
    result = subprocess.run(
        f"python -m common.list_models -p {p_value}",
        shell=True,
        capture_output=True,
        text=True
    )
    MODELS = result.stdout.strip().splitlines()
    ALIASES = ["Model A", "Model B", "Model C", "Model D"]

    # Print selected models
    print("Models to be used:")
    for i, model in enumerate(MODELS, start=1):
        alias = ALIASES[i-1] if i-1 < len(ALIASES) else f"Model {i}"
        print(f"  {alias}: {model}")

    print("Options:")
    print(f"  -p {p_value}")
    if v_flag:
        print(f"  {v_flag}")
    if n_value:
        print(f"  -n {n_value}")
    if kill_on_warning:
        print("  -k (kill on warning enabled)")

    # Step 1: Parse PDF
    print("Step 1: Convert PDF to JSON")
    run("python -m common.simple_parse", kill_on_warning=kill_on_warning)

    print("Step 2: Generate MCQs")
    run(f"python -m mcq_workflow.generate_mcqs -p {p_value} {v_flag}", kill_on_warning=kill_on_warning)

    print("Step 3: Combine JSON files")
    run("python -m common.combine_json_files -o MCQ-combined.json", kill_on_warning=kill_on_warning)

    if n_value:
        print(f"Selecting {n_value} MCQs at random...")
        run(f"python -m mcq_workflow.select_mcqs_at_random -i MCQ-combined.jsonl -o MCQ-subset.jsonl -n {n_value}", kill_on_warning=kill_on_warning)
        input_file = "MCQ-subset.json"
    else:
        input_file = "MCQ-combined.json"

    print("Step 4: Generate answers (all models)")
    model_names = ", ".join(MODELS)
    print(f"Generating answers with models: {model_names}")

    processes = []
    for model in MODELS:
        cmd = f"python -m mcq_workflow.generate_answers -i {input_file} -m {model} -p {p_value} {v_flag}"
        processes.append(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    for p in processes:
        for line in p.stdout:
            print(line, end='')
            if kill_on_warning and ("WARNING:" in line or "warning:" in line):
                print(f"Detected WARNING, killing workflow: {line.strip()}")
                for proc in processes:
                    proc.terminate()
                sys.exit(1)
        p.wait()

    print("Step 5: Score answers between all models")
    processes = []
    for i, model_a in enumerate(MODELS, start=1):
        for j, model_b in enumerate(MODELS, start=1):
            if i != j:
                print(f"Scoring {ALIASES[i-1]} answers using {ALIASES[j-1]}...")
                cmd = f"python -m mcq_workflow.score_answers -a {model_a} -b {model_b} -p {p_value} {v_flag}"
                processes.append(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    for p in processes:
        for line in p.stdout:
            print(line, end='')
            if kill_on_warning and ("WARNING:" in line or "warning:" in line):
                print(f"Detected WARNING, killing workflow: {line.strip()}")
                for proc in processes:
                    proc.terminate()
                sys.exit(1)
        p.wait()

    end_time = time.time()
    elapsed = int(end_time - start_time)
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total elapsed time: {hours:02}:{minutes:02}:{seconds:02} (hh:mm:ss)")


if __name__ == "__main__":
    main()

