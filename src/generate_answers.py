#!/usr/bin/env python

# generate_answers.py

import sys
import json
import os
import statistics
import requests
import time
import argparse
import logging

import config
from model_access import Model
from tqdm import tqdm

"""
import hydra
from omegaconf import DictConfig

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(cfg.general.noise_threshold)  # Accessing config values

if __name__ == "__main__":
    main()
"""


# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Program to use LLM to provide answers to MCQs')
    parser.add_argument('-m','--model', help='Model to use',
                        default=config.defaultModel)
    parser.add_argument('-s','--start', help='Number to start at in MCQs file',
                        default='0')
    parser.add_argument('-e','--end', help='End number in MCQs file',
                        default='all')
    parser.add_argument('-c', "--cache-dir", type=str,
                        default=os.getenv("HF_HOME"),
                        help="Custom cache directory for Hugging Face")
    parser.add_argument('-i', '--input',  help='file containing MCQs',
                        required=True)  
    parser.add_argument('-o', '--output', help='Output directory for Results',
                        default=config.results_dir)
    parser.add_argument('-q','--quiet',   action='store_true',   
                        help='No progress bar or messages')
    parser.add_argument('-v','--verbose', action='store_true',    
                        help='Enable verbose logging')    

    args = parser.parse_args()

    # Decide logging level and whether to show a progress bar

    if args.verbose:
        config.logger.setLevel(logging.INFO)
        use_progress_bar = False
        config.logger.info("verbose mode")
    elif args.quiet:
        config.logger.setLevel(logging.CRITICAL)
        use_progress_bar = False
        config.logger.info("quiet mode")
    else:  # default case
        config.logger.setLevel(logging.WARNING)
        use_progress_bar = True
        config.logger.info("progress bar only mode")

    # Set HF_HOME if using custom cache directory
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        config.logger.info(f"Using Hugging Face cache directory: {args.cache_dir}")

    json_file = args.input
    output_dir   = args.output

    model_name = args.model

    model = Model(model_name)
    model.details()

    # Load question-answer pairs
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        config.logger.error(f"ERROR: file {json_file} not found.")
        sys.exit(0)

    qa_pairs = []

    start_time = time.time()
    total_time = 0

    if args.end == 'all':
        data = data[int(args.start):]
    else:
        data = data[int(args.start):int(args.end)]

    config.logger.info(f'Generating {len(data)} answers with model {model_name}')

    # Create progress bar if not in --quiet or --verbose mode.
    if use_progress_bar:
        pbar = tqdm(total=len(data), desc="Processing", unit="item")
    else:
        pbar = config.NoOpTqdm(total=len(data))

    # Use JSON Lines file format, append each result immediately rather
    # than waiting to write everything at the end
    # Determine the output file name (changed extension to .jsonl).
    if int(args.start) == 0 and args.end == 'all':
        output_file = f'answers_{model_name.replace("/","+")}.jsonl'
    else:
        output_file = f'answers_{model_name.replace("/","+")}_{args.start}_{args.end}.jsonl'
    output_path = os.path.join(output_dir, output_file)

    # Remove existing output file if present; start fresh
    if os.path.exists(output_path):
        os.remove(output_path)

    for (qa_pair, index) in zip(data, range(1, len(data) + 1)):
        question = qa_pair.get("question", "")
        reference_answer = qa_pair.get("answer", "")
        filename         = qa_pair.get("file", "")
        filenum          = qa_pair.get("filenum", "")
        chunknum         = qa_pair.get("chunknum", "")

        if not question or not reference_answer:
            config.logger.info("not a question or ref_answer")
            continue  # skip malformed items

        # Use the model to generate an answer
        try:
            model_answer = model.run(question)
        except KeyboardInterrupt:
            config.logger.warning("EXIT: Execution interrupted by user")
            sys.exit(0)
        except Exception as e:
            config.logger.error(f"ERROR: {e}")
            sys.exit(0)

        gen_time    = time.time() - start_time
        total_time += gen_time
        start_time  = time.time()
        if index%10==0:
            avg_time = total_time / index  # Average time per item so far
            config.logger.info(f'{index} ({avg_time:.2f} s)', end =' ', flush=True) 

        new_tuple = {'file':filename, 'filenum':filenum, 'chunknum':chunknum,
                     'gen_time': f'{gen_time:.3f}',
                     'question':question, 'reference': reference_answer,
                     'model': model_answer}
        qa_pairs.append(new_tuple)


        # Append the new result immediately to the output file.

        with open(output_path, 'a', encoding='utf-8') as out_f:
            out_f.write(json.dumps(new_tuple, ensure_ascii=False) + "\n")

        pbar.update(1)
    
    pbar.close()

    config.logger.info("Processing complete")


if __name__ == "__main__":
    main()

