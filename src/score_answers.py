#!/usr/bin/env python

# parallel_score_answers.py

import sys
import json
import os
import time
import argparse
import statistics
import logging
import concurrent.futures
from tqdm import tqdm

import config
from model_access import Model

def score_answer(index, model, question: str, reference_answer: str, user_answer: str) -> float:
    """
    Calls the model to evaluate how consistent the user_answer is with the reference_answer.
    Returns a numeric score (float) from 1 to 10.
    """
    eval_prompt = config.score_main_prompt.format(
        question=question,
        reference_answer=reference_answer,
        user_answer=user_answer
    )
    system_msg = config.score_main_system

    response = model.run(
        user_prompt=eval_prompt,
        system_prompt=system_msg,
        temperature=0.0
    )

    try:
        score = float(response)
    except ValueError:
        score = try_again_to_extract_score(model, user_answer)

    return score

def try_again_to_extract_score(model, user_answer: str) -> float:
    """
    Attempts to parse out a final numeric answer using the fallback prompt.
    If that fails, returns 0.0.
    """
    fallback_prompt_text = config.score_fallback_prompt.format(user_answer=user_answer)
    fallback_system_text = config.score_fallback_system

    try:
        response = model.run(
            user_prompt=fallback_prompt_text,
            system_prompt=fallback_system_text,
            temperature=0.0
        )
        score = float(response)
    except Exception:
        config.logger.info(
            f"Score of 0 for bad response++++\n{user_answer}\n+++++++++\n"
        )
        score = 0.0

    return score

def process_qa_pair(index, qa_pair, modelB, modelA_name, modelB_name):
    """
    Process a single QA pair:
      - Extract fields from the QA pair.
      - Evaluate the answer (using score_answer) and measure evaluation time.
      - Return a tuple (index, result_dict, eval_time).
    If any required field is missing, logs an error and returns None.
    """
    question         = qa_pair.get("question", "")
    reference_answer = qa_pair.get("reference", "")
    model_answer     = qa_pair.get("model", "")
    gen_time         = qa_pair.get("gen_time", "")
    file             = qa_pair.get("file", "")
    filenum          = qa_pair.get("filenum", "")
    chunknum         = qa_pair.get("chunknum", "")

    if not question or not reference_answer or not model_answer:
        config.logger.error(f"Bad item at index {index}: missing question, reference, or model answer.")
        return None

    start_eval = time.time()
    score = score_answer(index, modelB, question, reference_answer, model_answer)
    eval_time = time.time() - start_eval

    result = {
        'modelA': modelA_name,
        'modelB': modelB_name,
        'index': index,
        'question': question,
        'reference': reference_answer,
        'model': model_answer,
        'score': score,
        'gen_time': gen_time,
        'eval_time': f'{eval_time:.4f}',
        'file': file,
        'filenum': filenum,
        'chunknum': chunknum
    }
    return (index, result, eval_time)

def main():
    parser = argparse.ArgumentParser(
        description='Use LLM B to rate answers provided by LLM A in parallel'
    )
    parser.add_argument('-a','--modelA_name',
                        help='Model A name', default=config.model["name"])
    parser.add_argument('-b','--modelB_name',
                        help='Model B name', default=config.model_b["name"])
    parser.add_argument('-o','--output',
                        help='Output directory', default=config.results_dir)
    parser.add_argument('-f','--force',
                        help='Process even if score file exists', action="store_true")
    parser.add_argument('-c', "--cache-dir", type=str,
                        default=os.getenv("HF_HOME"),
                        help="Custom cache directory for Hugging Face")
    parser.add_argument('-q','--quiet', action='store_true',
                        help='No progress bar or messages')
    parser.add_argument('-v','--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('-p','--parallel', type=int, default=4,
                        help='Number of parallel threads (default: 4)')
    args = parser.parse_args()

    # Set logging level and progress bar usage
    if args.verbose:
        config.logger.setLevel(logging.INFO)
        use_progress_bar = False
    elif args.quiet:
        config.logger.setLevel(logging.CRITICAL)
        use_progress_bar = False
    else:
        config.logger.setLevel(logging.WARNING)
        use_progress_bar = True

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        config.logger.info(f"Using Hugging Face cache directory: {args.cache_dir}")

    output_dir = args.output
    modelA_name = args.modelA_name
    modelB_name = args.modelB_name
    modelB = Model(modelB_name)

    # Load previously generated answers from Model A
    #answer_file = os.path.join(output_dir, 'answers_' + modelA_name.replace('/', '+') + '.json')
    # we are using jsonl now
    answer_file = os.path.join(output_dir, 'answers_' + modelA_name.replace('/', '+') + '.jsonl')

    config.logger.info(f'Looking for {answer_file}')
    if not os.path.exists(answer_file):
        config.logger.error(f'No answers file for {modelA_name}')
        config.initiate_shutdown("Initiating shutdown.")
        #sys.exit(1)

    score_file = os.path.join(
        output_dir,
        f'scores_{modelA_name.replace("/","+")}={modelB_name.replace("/","+")}.json'
    )
    if os.path.exists(score_file) and not args.force:
        config.logger.error(f"Score file already exists: {score_file}")
        config.initiate_shutdown("Initiating shutdown.")
        #sys.exit(1)

    #with open(answer_file, "r", encoding="utf-8") as f:
    #    data = json.load(f)
    with open(answer_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]


    total_items = len(data)
    if use_progress_bar:
        pbar = tqdm(total=total_items, desc="Processing", unit="item")
    else:
        from config import NoOpTqdm
        pbar = NoOpTqdm(total=total_items)

    # We'll accumulate results in a buffer and write them periodically.
    #SAVE_INTERVAL = 10
    SAVE_INTERVAL = config.saveInterval # for simplicty, use the same value as for parallel_generate_answers
    output_buffer = []
    scores_list = []
    eval_total_time = 0.0
    processed_count = 0

    config.logger.info(f'Processing {total_items} QA pairs')

    # Open output file for writing (we'll write in JSON Lines format)
    if os.path.exists(score_file):
        os.remove(score_file)
    out_f = open(score_file, 'a', encoding='utf-8')

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        future_to_index = {
            executor.submit(process_qa_pair, idx, qa_pair, modelB, modelA_name, modelB_name): idx
            for idx, qa_pair in enumerate(data, start=1)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            res = future.result()
            if res is None:
                # Skip if item was bad
                processed_count += 1
                pbar.update(1)
                continue
            idx, result, eval_time = res
            output_buffer.append(result)
            scores_list.append(result['score'])
            eval_total_time += eval_time
            processed_count += 1

            # Log average times every SAVE_INTERVAL items
            if processed_count % SAVE_INTERVAL == 0:
                avg_eval_time = eval_total_time / processed_count
                config.logger.info(f"{processed_count} items processed (avg eval time = {avg_eval_time:.2f}s)")
                # Write out buffered results
                for item in output_buffer:
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                out_f.flush()
                output_buffer = []
            pbar.update(1)

    pbar.close()
    # Write any remaining items in the output buffer.
    if output_buffer:
        for item in output_buffer:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()

    if scores_list:
        mean_score = statistics.mean(scores_list)
        variance_score = statistics.pvariance(scores_list)
        config.logger.info(f"Scores computed: mean = {mean_score:.2f}, variance = {variance_score:.2f}")
    else:
        config.logger.warning("No valid QA pairs found or no scores computed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        #config.logger.warning("EXIT: Execution interrupted by user")
        config.initiate_shutdown("User interrupt - initiating shutdown.")
        #sys.exit(0)

