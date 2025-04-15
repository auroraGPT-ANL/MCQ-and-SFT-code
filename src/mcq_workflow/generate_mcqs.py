#!/usr/bin/env python
"""
generate_mcqs.py

Main entry point for generating MCQs in parallel.
Parses command-line arguments, initializes the model,
and then calls process_directory from mcq_util.
"""

import os
import argparse
from common import config
from common.model_access import Model
from .mcq_util import process_directory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate MCQs from JSON/JSONL files in parallel'
    )
    parser.add_argument('-i', '--input', help='Directory containing input JSON/JSONL files', default=config.json_dir)
    parser.add_argument('-o', '--output', help='Output directory for MCQs', default=config.mcq_dir)
    parser.add_argument('-m', '--model', help='Model to use to generate MCQs', default=config.defaultModel)
    parser.add_argument('-q', '--quiet', action='store_true', help='No progress bar or messages')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('-p', '--parallel', type=int, default=4, help='Number of parallel threads (default: 4)')
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if output files exist.')

    args = parser.parse_args()

    use_progress_bar = config.configure_verbosity(args)
    input_directory = args.input
    output_directory = args.output
    model_name = args.model

    model = Model(model_name)
    model.details()

    os.makedirs(output_directory, exist_ok=True)

    try:
        process_directory(model, input_directory, output_directory,
                          use_progress_bar=use_progress_bar,
                          parallel_workers=args.parallel,
                          force=args.force)
    except KeyboardInterrupt:
        config.initiate_shutdown("User Interrupt - initiating shutdown.")

