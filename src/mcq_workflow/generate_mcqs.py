#!/usr/bin/env python
"""
generate_mcqs.py

Main entry point for generating MCQs in parallel.
Parses command-line arguments, initializes the model,
then calls process_directory from mcq_util.
"""

import os
import argparse
from types import SimpleNamespace
from common import config
from common.model_access import Model
from .mcq_util import process_directory


def generate_mcqs_dir(input_dir: str,
                       output_dir: str,
                       model_name: str,
                       parallel_workers: int = 4,
                       verbose: bool = False,
                       force: bool = False) -> str:
    """
    Generate MCQs from JSON/JSONL files.

    Args:
      input_dir: directory containing input JSON/JSONL files
      output_dir: directory to write MCQ files (JSONL)
      model_name: model to use for generation
      parallel_workers: number of parallel threads
      verbose: enable verbose logging
      force: force reprocessing even if outputs exist

    Returns:
      The path to the directory where MCQs were written.
    """
    # Configure verbosity (only set verbose or quiet, not both)
    dummy_args = SimpleNamespace(verbose=verbose, quiet=False)
    use_progress_bar = config.configure_verbosity(dummy_args)

    # Initialize and display model details
    model = Model(model_name, parallel_workers=parallel_workers)

    model.details()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Run the main MCQ-generation logic
        process_directory(
            model,
            input_dir,
            output_dir,
            use_progress_bar=use_progress_bar,
            parallel_workers=parallel_workers,
            force=force,
        )
    except KeyboardInterrupt:
        config.initiate_shutdown("User Interrupt - initiating shutdown.")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate MCQs from JSON/JSONL files in parallel'
    )
    parser.add_argument(
        '-i', '--input',
        help='Directory containing input JSON/JSONL files',
        default=config.json_dir
    )
    parser.add_argument(
        '-o', '--output',
        help='Output directory for MCQs',
        default=config.mcq_dir
    )
    parser.add_argument(
        '-m', '--model',
        help='Model to use to generate MCQs',
        default=config.defaultModel
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='No progress bar or messages'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '-p', '--parallel',
        type=int,
        default=4,
        help='Number of parallel threads (default: 4)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing files to generate (and append) MCQs even if output files exist.'
    )

    args = parser.parse_args()

    # Call the new Python API and capture the output directory
    mcq_outdir = generate_mcqs_dir(
        input_dir=args.input,
        output_dir=args.output,
        model_name=args.model,
        parallel_workers=args.parallel,
        verbose=args.verbose,
        force=args.force,
    )
    # Print the resulting path for wrappers or agent capture
    print(mcq_outdir)
