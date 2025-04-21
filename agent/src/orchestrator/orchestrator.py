#!/usr/bin/env python
"""
Orchestrator for the Agent‑based MCQ workflow.

Reads only the two flags (parallel threads, verbose) plus force options,
uses defaults from common.config for all paths and models,
and invokes each Agent in Pipeline in order.
"""
import argparse
import os

from common import config
from orchestrator.registry import Pipeline

# Auto‑detect project root relative to this file
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..'))


def main():
    parser = argparse.ArgumentParser(
        description="Run the MCQ agent pipeline using config.yml defaults"
    )
    parser.add_argument(
        "-p", "--parallel", type=int,
        default=config.defaultThreads,
        help="Number of parallel threads for generation/answering/scoring"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--force_mcq", action="store_true",
        help="Regenerate MCQs even if outputs already exist"
    )
    parser.add_argument(
        "--force_score", action="store_true",
        help="Rescore answers even if outputs already exist"
    )
    args = parser.parse_args()

    # Build initial context
    ctx = {
        "project_root": PROJECT_ROOT,
        "parsed_dir":   config.json_dir,     # default from config
        "mcq_out":      config.mcq_dir,      # default from config
        "combined_mcq": os.path.join(PROJECT_ROOT, "MCQ-combined.jsonl"),
        "answer_dir":   config.results_dir,  # default from config
        "score_dir":    config.results_dir,  # same as answers
        "models":       [config.model["name"], config.model_b["name"]],
        "p_value":      args.parallel,
        "v_flag":       args.verbose,
        "force_mcq":    args.force_mcq,
        "force_score":  args.force_score,
    }

    # Run each agent
    for agent in Pipeline:
        print(f"▶️  Running {agent.__class__.__name__} …")
        outputs = agent.run(ctx)
        ctx.update(outputs)

    # Final summary
    print("✅ Pipeline complete. Summary:")
    print(f"  parsed_dir:   {ctx['parsed_dir']}")
    print(f"  mcq_out:      {ctx['mcq_out']}")
    print(f"  combined_mcq: {ctx['combined_mcq']}")
    print(f"  answer_file:  {ctx.get('answer_file')}" )
    print(f"  score_file:   {ctx.get('score_file')}" )
    print(f"  force_mcq:    {ctx['force_mcq']}")
    print(f"  force_score:  {ctx['force_score']}")

if __name__ == "__main__":
    main()

