#!/usr/bin/env python

# generate_nuggets.py

import os
import argparse
from common import config
from common.model_access import Model
from .nugget_utils import process_directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate augmented chunks from JSONL/JSON files')
    parser.add_argument('-i', '--input', help='Directory containing input JSON/JSONL files', default=config.json_dir)
    parser.add_argument('-o', '--output',
                      help='Output JSONL file (default: nuggets.jsonl in results directory)',
                      default=os.path.join(config.results_dir, 'nuggets.jsonl'))
    parser.add_argument('-m', '--model', help='Model to use', default=config.defaultModel)
    parser.add_argument('-q', '--quiet', action='store_true', help='No progress bar or messages')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('-p', '--parallel', type=int, default=4, help='Number of parallel threads (default: 4)')

    args = parser.parse_args()

    use_progress_bar = config.configure_verbosity(args)
    input_directory = args.input
    output_file = args.output
    model_name = args.model
    model = Model(model_name)
    model.details()

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        process_directory(model, input_directory, output_file,
                          use_progress_bar=use_progress_bar,
                          parallel_workers=args.parallel)
    except KeyboardInterrupt:
        config.initiate_shutdown("User Interrupt - initiating shutdown.")

