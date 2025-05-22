#!/usr/bin/env python

# generate_nuggets.py

import os
import argparse
from common import config  # Keep for backward compatibility
from common.settings import load_settings
from common.model_access import Model
from nugget_workflow.nugget_utils import process_directory

# Initialize settings
settings = load_settings()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate augmented chunks from JSONL/JSON files')

    # Get directory values from settings, fall back to config for backward compatibility
    json_dir = settings.directories.json_dir if hasattr(settings, 'directories') else config.json_dir
    results_dir = settings.directories.results if hasattr(settings, 'directories') else config.results_dir

    # Get extraction model from settings workflow, fall back to config.defaultModel
    default_model = settings.workflow.extraction if hasattr(settings, 'workflow') else config.defaultModel

    parser.add_argument('-i', '--input', help='Directory containing input JSON/JSONL files', default=json_dir)
    parser.add_argument('-o', '--output',
                      help='Output JSONL file (default: nuggets.jsonl in results directory)',
                      default=os.path.join(results_dir, 'nuggets.jsonl'))
    parser.add_argument('-m', '--model', help='Model to use', default=default_model)
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
        # Use config.initiate_shutdown for compatibility
        config.initiate_shutdown("User Interrupt - initiating shutdown.")
    except Exception as e:
        config.logger.error(f"Error processing directory: {e}")
        config.initiate_shutdown(f"Error during processing: {e}")

