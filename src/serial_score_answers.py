#!/usr/bin/env python

# score_answers.py

import sys
import json
import os
import statistics
import time
import argparse
from tqdm import tqdm
import logging
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
    If that fails, returns 0.0
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
    except:
        config.logger.info(
            f"Score of 0 for bad response++++\n{user_answer}\n+++++++++\n"
        )
        score = 0.0

    return score


def main():
    parser = argparse.ArgumentParser(
        description='Program to use LLM B to rate answers provided previously by LLM A'
    )
    parser.add_argument( '-a','--modelA_name',
                        help='modelA name', default=config.model["name"])
    parser.add_argument( '-b','--modelB_name',
                        help='modelB name', default=config.model_b["name"])
    parser.add_argument('-o','--output',
                        help='Output directory', default=config.results_dir)
    parser.add_argument('-f','--force',
                        help='Process even if score file exists', action="store_true")
    parser.add_argument('-c', "--cache-dir", type=str,
                        default=os.getenv("HF_HOME"),
                        help="Custom cache directory for Hugging Face")
    parser.add_argument('-q','--quiet',   action='store_true',
                        help='No progress bar or messages')
    parser.add_argument('-v','--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    use_progress_bar = config.configure_verbosity(args)

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        config.logger.info(f"Using Hugging Face cache directory: {args.cache_dir}")

    output_dir = args.output

    modelA_name = args.modelA_name
    modelB_name = args.modelB_name
    modelB = Model(modelB_name)

    # Load previously generated answers from modelA
    #answer_file = os.path.join(
    #    output_dir,
    #    'answers_' + modelA_name.replace('/', '+') + '.json'
    #)
    # we are using jsonl now
    answer_file = os.path.join(output_dir, 'answers_' + modelA_name.replace('/', '+') + '.jsonl')

    config.logger.info(f'Looking for {answer_file}')
    if not os.path.exists(answer_file):
        config.logger.error(f'No answers file for {modelA_name}')
        sys.exit(1)

    score_file = os.path.join(
        output_dir,
        f'scores_{modelA_name.replace("/","+")}={modelB_name.replace("/","+")}.json'
    )
    if os.path.exists(score_file) and not args.force:
        config.logger.error(f"Score file already exists: {score_file}")
        sys.exit(1)

# we are now using jsonl (json lists) rather than json in our outputs from
# generate_mcqs and generate_answers (and the parallel versions)

    #with open(answer_file, "r", encoding="utf-8") as f:
        #data = json.load(f)
    with open(answer_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    from config import NoOpTqdm
    if use_progress_bar:
        from tqdm import tqdm
        pbar = tqdm(total=len(data), desc="Processing", unit="item")
    else:
        pbar = NoOpTqdm(total=len(data))

    scores   = []
    qa_pairs = []
    import time
    start_time = time.time()
    total_time = 0
    eval_answer_total_time = 0

    config.logger.info(f'Processing {len(data)} QA pairs')
    out_path = score_file
    out_f = open(out_path, 'w', encoding='utf-8')

    for (qa_pair, index) in zip(data, range(1, len(data) + 1)):
        question         = qa_pair.get("question", "")
        reference_answer = qa_pair.get("reference", "")
        model_answer     = qa_pair.get("model", "")
        gen_time         = qa_pair.get("gen_time", "")
        file             = qa_pair.get("file", "")
        filenum          = qa_pair.get("filenum", "")
        chunknum         = qa_pair.get("chunknum", "")

        if not question or not reference_answer or not model_answer:
            config.logger.error("Bad item:")
            config.logger.error(f"question: {question}")
            config.logger.error(f"reference: {reference_answer}")
            config.logger.error(f"model: {model_answer}")
            sys.exit(1)

        eval_answer_start_time = time.time()
        score = score_answer(index, modelB, question, reference_answer, model_answer)
        eval_answer_time = time.time() - eval_answer_start_time
        eval_answer_total_time += eval_answer_time

        if score is not None:
            scores.append(score)
            qa_pairs.append({
                'modelA': modelA_name,
                'modelB': modelB_name,
                'index': index,
                'question': question,
                'reference': reference_answer,
                'model': model_answer,
                'score': score,
                'gen_time': gen_time,
                'eval_time': f'{eval_answer_time:.4f}',
                'file': file,
                'filenum': filenum,
                'chunknum': chunknum
            })

        total_time += time.time() - start_time
        start_time = time.time()

        if index % 10 == 0:
            avg_time = total_time / index
            avg_eval_time = eval_answer_total_time / index
            config.logger.info(
                f"{index} (avg_time={avg_time:.2f}s, eval_time={avg_eval_time:.2f}s)"
            )

        pbar.update(1)

    pbar.close()

    json.dump(qa_pairs, out_f, ensure_ascii=False, indent=2)
    out_f.close()

    if scores:
        import statistics
        mean_score = statistics.mean(scores)
        variance_score = statistics.pvariance(scores)
        config.logger.info(f"Scores computed: mean={mean_score:.2f}, variance={variance_score:.2f}")
    else:
        config.logger.warning("No valid QA pairs found or no scores computed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        config.logger.warning("EXIT: Execution interrupted by user")
        sys.exit(0)

