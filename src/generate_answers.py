#!/usr/bin/env python

# parallel_generate_answers.py

import sys
import json
import os
import time
import argparse
import logging
import concurrent.futures

import config
from model_access import Model
from tqdm import tqdm

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
    
    new_tuple = {
        'file': filename,
        'filenum': filenum,
        'chunknum': chunknum,
        'gen_time': f'{gen_time:.3f}',
        'question': question,
        'reference': reference_answer,
        'model': model_answer
    }
    return (index, new_tuple)

def main():
    parser = argparse.ArgumentParser(description='Use LLM to provide answers to MCQs in parallel, saving periodically.')
    parser.add_argument('-m','--model', help='Model to use',
                        default=config.defaultModel)
    parser.add_argument('-s','--start', help='Start index in MCQs file (default: 0)',
                        default='0')
    parser.add_argument('-e','--end', help='End index in MCQs file (or "all")',
                        default='all')
    parser.add_argument('-c', "--cache-dir", type=str,
                        default=os.getenv("HF_HOME"),
                        help="Custom cache directory for Hugging Face")
    parser.add_argument('-i', '--input',  help='File containing MCQs',
                        required=True)
    parser.add_argument('-o', '--output', help='Output directory for results',
                        default=config.results_dir)
    parser.add_argument('-p', '--parallel', type=int, default=4,
                        help='Number of parallel threads (default: 4)')
    parser.add_argument('-q','--quiet',   action='store_true',
                        help='No progress bar or messages')
    parser.add_argument('-v','--verbose', action='store_true',
                        help='Enable verbose logging')
    args = parser.parse_args()

    use_progress_bar = config.configure_verbosity(args)

    # Set Hugging Face cache directory if provided
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        config.logger.info(f"Using Hugging Face cache directory: {args.cache_dir}")

    json_file = args.input
    output_dir = args.output

    model_name = args.model
    model = Model(model_name)
    model.details()

    # Load QA pairs from input file
    try:
        data = []
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    config.logger.warning(f"Skipping invalid JSON line: {e}")
                    continue
    except Exception as e:
        config.logger.error(f"ERROR: File {json_file} not found or could not be read: {e}")
        config.initiate_shutdown("Initiating shutdown.")
        #sys.exit(0)

    if not data:
        config.logger.error("No valid data found in input file")
        config.initiate_shutdown("Initiating shutdown.")
        #sys.exit(0)

    start_index = int(args.start)
    if args.end == 'all':
        data = data[start_index:]
    else:
        end_index = int(args.end)
        data = data[start_index:end_index]
    
    total_items = len(data)
    config.logger.info(f"Generating answers for {total_items} items using model {model_name}")

    # Determine output file name
    if start_index == 0 and args.end == 'all':
        output_file = f'answers_{model_name.replace("/","+")}.jsonl'
    else:
        output_file = f'answers_{model_name.replace("/","+")}_{args.start}_{args.end}.jsonl'
    output_path = os.path.join(output_dir, output_file)

    # Remove existing output file if present
    if os.path.exists(output_path):
        os.remove(output_path)

    # Save results periodically every SAVE_INTERVAL items.
    #SAVE_INTERVAL = 50
    SAVE_INTERVAL = config.saveInterval
    config.logger.info(f"Write to results file every {SAVE_INTERVAL} QA's processed.")
    output_buffer = []

    if use_progress_bar:
        pbar = tqdm(total=total_items, desc="Processing", unit="item")
    else:
        pbar = config.NoOpTqdm(total=total_items)

    # Open the output file once and append to it periodically.
    with open(output_path, 'a', encoding='utf-8') as out_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
            future_to_index = {
                executor.submit(process_qa_item, qa_item, idx, model): idx 
                for idx, qa_item in enumerate(data, start=1)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                idx, result = future.result()
                if result is not None:
                    output_buffer.append(result)
                pbar.update(1)
                # If we've collected SAVE_INTERVAL items, write them to file.
                if len(output_buffer) >= SAVE_INTERVAL:
                    for item in output_buffer:
                        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    out_f.flush()
                    output_buffer = []
        # Write any remaining results in the buffer.
        if output_buffer:
            for item in output_buffer:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            out_f.flush()
    pbar.close()
    config.logger.info(f"Output written to {output_path}")
    config.logger.info("Processing complete")

if __name__ == "__main__":
    main()

