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
import json
import concurrent.futures
from typing import Union, List
from tqdm import tqdm

from common import config
from common.model_access import Model


def process_mcq_item(mcq_item, index, model):
    """
    Process a single QA pair:
      - Extract the question and multiple choice answers
      - Use the model to select one f the answers
      - Compute the generation time.
      - Return a tuple (index, result) with the updated MCQ data.
    """

    question = mcq_item.get("question", "")
    choices = mcq_item.get("choices", "")
    reference_answer = mcq_item.get("reference_answer", "")
    filename = mcq_item.get("file", "")
    filenum = mcq_item.get("filenum", "")
    chunknum = mcq_item.get("chunknum", "")

    if not question or not choices or not reference_answer:
        config.logger.info(f"Item {index} missing question, choices, or reference answer; skipping.")
        return (index, None)

    start_time = time.time()

    user_message = config.user_message_mcq_answer.format(num_answers=7, question=question, choices=choices)

    try:
        #print('AAA', user_message)
        #print('BBB', config.system_message_mcq_answer)
        model_answer = model.run(user_prompt=user_message, system_prompt=config.system_message_mcq_answer)
        #print(f'XXX-{model_answer}-XXX')
        cleaned = model_answer.replace("```json", "").replace("```", "").strip()
        model_answer2 = json.loads(cleaned)
        answer = model_answer2['answer']
        #comment = model_answer2['comment']
    except Exception as e:
        config.logger.error(f"Error processing item {index}: {e}")
        #print('ERROR', model_answer)
        #exit(1)
        return (index, None)
    gen_time = time.time() - start_time

    result = {
        'file': filename,
        'filenum': filenum,
        'chunknum': chunknum,
        'gen_time': f'{gen_time:.3f}',
        'question': question,
	'choices': choices,
        'model_answer': answer,
        'reference_answer': reference_answer,
        'answers_match': answer == int(reference_answer),
        #'comment': comment,
        'model': f'{model.model_type}:{model.model_name}'
    }
    return (index, result)


def generate_answers_file(
    input_file: str,
    model_name: str = config.defaultModel,
    output_dir: str = None,
    parallel: int = None,
    cache_dir: str = None,
    quiet: bool = False,
    verbose: bool = False,
    force: bool = False
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

    # Prepare output path
    out_dir = output_dir or config.results_dir
    os.makedirs(out_dir, exist_ok=True)
    fname = f"answers_{model_name.replace('/','+')}.jsonl"
    output_path = os.path.join(out_dir, fname)

    if not force and os.path.exists(output_path):
        print(f"Skipping as already exists: {output_path}")
        config.logger.info(f"Skipping as already exists: {output_path}")
        return

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
        #raise ValueError(f"Input file {input_file} is empty.")
        print(f"Input file {input_file} is empty.")
        return
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

    total = len(data)
    config.logger.info(f"Generating answers for {total} items with model {model_name}")

    # Determine thread count
    max_workers = parallel or config.defaultThreads

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
                executor.submit(process_mcq_item, mcq, idx, model): idx
                for idx, mcq in enumerate(data, start=1)
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
    parser.add_argument('-c','--cache-dir',default=os.getenv('HF_HOME'))
    parser.add_argument('-i','--input',    default=None, help='JSON or JSONL file')
    parser.add_argument('-o','--output',   default=None)
    parser.add_argument('-p','--parallel', type=int, default=config.defaultThreads)
    parser.add_argument('-q','--quiet',    action='store_true')
    parser.add_argument('-v','--verbose',  action='store_true')
    parser.add_argument('-f','--force',    action='store_true')
    args = parser.parse_args()

    try:
        out_file = generate_answers_file(
            input_file   = args.input,
            model_name   = args.model,
            output_dir   = args.output,
            parallel     = args.parallel,
            cache_dir    = args.cache_dir,
            quiet        = args.quiet,
            verbose      = args.verbose,
            force        = args.force
        )
        print(out_file)
    except Exception as e:
        logging.error(str(e))
        sys.exit(1)

if __name__ == '__main__':
    main()

