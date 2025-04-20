#!/usr/bin/env python
"""
Orchestrator for the Agent‑based MCQ workflow.
"""

import argparse
import os
from orchestrator.registry import Pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Run the MCQ agent pipeline end‑to‑end"
    )
    parser.add_argument(
        "--project_root", default=os.getcwd(),
        help="Root directory of the project"
    )
    parser.add_argument(
        "-p", "--threads", type=int, default=4,
        help="Number of parallel threads for MCQ generation"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--parsed_dir", default=None,
        help="If you already ran parsing, point here to skip that step"
    )
    args = parser.parse_args()

    # Build the initial context
    ctx = {
        "project_root": args.project_root,
        "p_value":      args.threads,
        "v_flag":       args.verbose,
    }
    if args.parsed_dir:
        ctx["parsed_dir"] = args.parsed_dir

    # Run each agent in order
    for agent in Pipeline:
        name = agent.__class__.__name__
        print(f"▶️  Running {name} …")
        outputs = agent.run(ctx)
        ctx.update(outputs)

    # Final report
    mcq_file = ctx.get("mcq_file")
    print(f"✅ Done. MCQs written to: {mcq_file}")

if __name__ == "__main__":
    main()

