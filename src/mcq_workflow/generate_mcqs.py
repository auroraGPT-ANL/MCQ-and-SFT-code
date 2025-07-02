#!/usr/bin/env python
"""
generate_mcqs.py

Main entry point for generating MCQs in parallel.
Parses command-line arguments, initializes the model,
then calls process_directory from mcq_util.
"""

import os
import argparse
import logging
import sys
from types import SimpleNamespace
from common import config  # Keep for backward compatibility
from mcq_workflow.mcq_util import process_directory

# -----------------------------------
# Logging: default WARNING unless -v/--verbose
level = logging.DEBUG if "-v" in sys.argv or "--verbose" in sys.argv else logging.WARNING
logging.basicConfig(format="%(levelname)s:%(name)s: %(message)s", level=level)
logging.getLogger("httpx").setLevel(level)
# -----------------------------------

from common.loader import load_settings
from common.model_access import Model

# Initialize settings
settings = load_settings()


def generate_mcqs_dir(
    input_dir: str,
    output_dir: str,
    model_name: str,
    parallel_workers: int = 4,
    verbose: bool = False,
    force: bool = False,
    num_answers: int = 4,
) -> str:
    """
    Generate MCQs from JSON/JSONL files.
    Returns the path to the directory where MCQs were written.
    """
    dummy_args = SimpleNamespace(verbose=verbose, quiet=False)
    use_progress_bar = config.configure_verbosity(dummy_args)

    model = Model(model_name, parallel_workers=parallel_workers)
    model.details()

    os.makedirs(output_dir, exist_ok=True)

    try:
        process_directory(
            model,
            input_dir,
            output_dir,
            use_progress_bar=use_progress_bar,
            parallel_workers=parallel_workers,
            force=force,
            num_answers=num_answers,
        )
    except KeyboardInterrupt:
        config.initiate_shutdown("User Interrupt - initiating shutdown.")
        sys.exit(1)

    return output_dir


def main():
    # Directory defaults
    json_dir = (
        settings.directories.json_dir
        if hasattr(settings, 'directories')
        else config.json_dir
    )
    mcq_dir = (
        settings.directories.mcq
        if hasattr(settings, 'directories')
        else config.mcq_dir
    )
    default_model = (
        settings.workflow.extraction
        if hasattr(settings, 'workflow')
        else config.defaultModel
    )

    parser = argparse.ArgumentParser(
        description='Generate MCQs from JSON/JSONL files in parallel'
    )
    parser.add_argument('-i', '--input', help='Input directory', default=json_dir)
    parser.add_argument('-o', '--output', help='Output directory', default=mcq_dir)
    parser.add_argument('-m', '--model', help='Model name', default=default_model)
    parser.add_argument('-q', '--quiet', action='store_true', help='No progress bar')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('-p', '--parallel', type=int, default=4, help='Number of threads')
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if outputs exist')
    parser.add_argument('-a', '--answers', type=int, default=4, help='Number of answers to generate')
    args = parser.parse_args()

    try:
        mcq_outdir = generate_mcqs_dir(
            input_dir=args.input,
            output_dir=args.output,
            model_name=args.model,
            parallel_workers=args.parallel,
            verbose=args.verbose,
            force=args.force,
            num_answers=args.answers,
        )
        print(mcq_outdir)
    except KeyboardInterrupt:
        config.initiate_shutdown("User Interrupt - initiating shutdown.")
        sys.exit(1)


if __name__ == '__main__':
    main()

