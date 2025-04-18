#!/usr/bin/env python
"""
generate_answers.py

Provides both a Python API and CLI for generating answers to MCQs in parallel.
"""
import os
import sys
import json
import time
import argparse
import logging
import concurrent.futures
from typing import Union, List
from tqdm import tqdm

from common import config
from common.model_access import Model


def process_qa_item(qa_item, index, model):
    """
    Process a single QA pair:
      - Extract the question and reference answer.
      - Use the model to generate an answer.
      - Compute the generation time.
      - Return a tuple (index, result) with the updated QA data.
    """
    question = qa_item.get("question", "")
    reference_answer = qa_item.get("answer", "")
    filename = qa_item.get("file", "")
    filenum = qa_item.get("filenum", "")
    chunknum = qa_item.get("chunknum", "")

    if not question or not reference_answer:
        config.logger.info(f"Item {index} missing question or reference answer; skipping.")
        return (index, None)

    start_time = time.time()
    try:
        model_answer = model.run(question)
    except Exception as e:
        config.logger.error(f"Error processing item {index}: {e}")
        return (index, None)
    gen_time = time.time() - start_time

    result = {
        'file': filename,
        'filenum': filenum,
        'chunknum': chunknum,
        'gen_time': f'{gen_time:.3f}',
        'question': question,
        'reference': reference_answer,
        'model': model_answer
    }
    return (index, result)


def generate_answers_file(
    input_file: str,
    model_name: str = config.defaultModel,
    output_dir: str = None,
    start_index: int = 0,
    end_index: Union[int, str] = 'all',
    parallel: int = None,
    cache_dir: str = None,
    quiet: bool = False,
    verbose: bool = False
) -> str:
    """
    Generate model answers for MCQs stored in a JSON or JSONL file.

    Returns:
      The path to the output JSONL file.
    """
    
    if input_file is None:
        # pick the first *.jsonl in config.mcq_dir
        jsonl_files = [f for f in os.listdir(config.mcq_dir) if f.endswith(".jsonl")]
        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files found in {config.mcq_dir}")
        input_file = os.path.join(config.mcq_dir, jsonl_files[0])
        config.logger.info(f"Using default MCQ file: {input_file}")


    # Configure verbosity

    args_ns = argparse.Namespace(quiet=quiet, verbose=verbose)
    use_progress_bar = config.configure_verbosity(args_ns)

    # Set cache directory if provided
    if cache_dir:
        os.environ['HF_HOME'] = cache_dir
        config.logger.info(f"Using cache directory: {cache_dir}")

    # Resolve input file path
    if not os.path.isabs(input_file) and not os.path.exists(input_file):
        input_file = os.path.join(config.mcq_dir, input_file)

    # Load data from JSON or JSONL
    data: List[dict] = []
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f"Input file {input_file} is empty.")
    if content[0] == '[':
        items = json.loads(content)
        for itm in items:
            if isinstance(itm, list):
                data.extend(itm)
            else:
                data.append(itm)
    else:
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                itm = json.loads(line)
                if isinstance(itm, list):
                    data.extend(itm)
                else:
                    data.append(itm)
            except json.JSONDecodeError:
                config.logger.warning(f"Skipping invalid JSON line: {line}")

    if not data:
        raise ValueError("No valid items found in input file.")

    # Slice data between start_index and end_index
    start = start_index
    if end_index == 'all':
        slice_data = data[start:]
    else:
        slice_data = data[start:int(end_index)]

    total = len(slice_data)
    config.logger.info(f"Generating answers for {total} items with model {model_name}")

    # Determine thread count
    max_workers = parallel or config.defaultThreads

    # Prepare output path
    out_dir = output_dir or config.results_dir
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{start}_{end_index}" if start != 0 or end_index != 'all' else ''
    fname = f"answers_{model_name.replace('/','+')}{suffix}.jsonl"
    output_path = os.path.join(out_dir, fname)

    # Remove existing output
    if os.path.exists(output_path):
        os.remove(output_path)

    # Execute in parallel
    buffer: List[dict] = []
    save_int = config.saveInterval
    model = Model(model_name)
    model.details()

    if use_progress_bar:
        pbar = tqdm(total=total, desc="Answering", unit="item")
    else:
        pbar = config.NoOpTqdm(total=total)

    with open(output_path, 'a', encoding='utf-8') as out_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_qa_item, qa, idx, model): idx
                for idx, qa in enumerate(slice_data, start=1)
            }
            for fut in concurrent.futures.as_completed(futures):
                idx, res = fut.result()
                if res:
                    buffer.append(res)
                pbar.update(1)
                if len(buffer) >= save_int:
                    for item in buffer:
                        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    out_f.flush()
                    buffer.clear()
        # Flush remaining
        for item in buffer:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        out_f.flush()
    pbar.close()
    config.logger.info(f"Answers written to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate answers for MCQs via LLM.'
    )
    parser.add_argument('-m','--model',    default=config.defaultModel)
    parser.add_argument('-s','--start',    type=int, default=0)
    parser.add_argument('-e','--end',      default='all')
    parser.add_argument('-c','--cache-dir',default=os.getenv('HF_HOME'))
    parser.add_argument('-i','--input',    default=None)
    parser.add_argument('-o','--output',   default=None)
    parser.add_argument('-p','--parallel', type=int, default=config.defaultThreads)
    parser.add_argument('-q','--quiet',    action='store_true')
    parser.add_argument('-v','--verbose',  action='store_true')
    args = parser.parse_args()

    try:
        out_file = generate_answers_file(
            input_file   = args.input,
            model_name   = args.model,
            output_dir   = args.output,
            start_index  = args.start,
            end_index    = args.end,
            parallel     = args.parallel,
            cache_dir    = args.cache_dir,
            quiet        = args.quiet,
            verbose      = args.verbose
        )
        print(out_file)
    except Exception as e:
        logging.error(str(e))
        sys.exit(1)

if __name__ == '__main__':
    main()

