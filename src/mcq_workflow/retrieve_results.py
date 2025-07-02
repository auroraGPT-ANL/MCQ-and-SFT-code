"""
retrieve_results.py

Provides both a Python API and CLI for reading results from a JSONL file and counting correct/incorrect
"""
import os
import sys
import json
import time
import argparse
import logging
import json
import concurrent.futures
from typing import Union, List
from tqdm import tqdm

from common import config

def retrieve_answers_file(
    input_file: str,
) -> str:
    """
    Retrieve model answers for MCQs

    Returns:
      Number of match/non-match
    """
    
    # Resolve input file path
    if not os.path.isabs(input_file) and not os.path.exists(input_file):
        input_file = os.path.join(config.results_dir, input_file)

    true_count = 0
    false_count = 0

    with open(input_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            if data['answers_match']:
                true_count += 1
            else:
                false_count += 1

    return (true_count, false_count)


def main():
    parser = argparse.ArgumentParser(
        description='Generate answers for MCQs via LLM.'
    )
    parser.add_argument('-i','--input',    required=True)
    args = parser.parse_args()

    try:
        (correct, wrong) = retrieve_answers_file(
            input_file   = args.input
        )
        print(f'{correct} correct, {wrong} wrong; {correct+wrong} total')
    except Exception as e:
        logging.error(str(e))
        sys.exit(1)

if __name__ == '__main__':
    main()

