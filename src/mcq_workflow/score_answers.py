#!/usr/bin/env python
"""
score_answers.py

Provides both a Python API and CLI for scoring answers produced by one model (A) using another model (B) in parallel.
"""
import os
import json
import time
import argparse
import statistics
import logging
import concurrent.futures
from typing import Optional, List, Tuple
from tqdm import tqdm
from common import config  # Keep for backward compatibility
import logging
import sys

# -----------------------------------
# Logging: default WARNING unless -v/--verbose
level = logging.DEBUG if "-v" in sys.argv or "--verbose" in sys.argv else logging.WARNING
logging.basicConfig(format="%(levelname)s:%(name)s: %(message)s", level=level)
logging.getLogger("httpx").setLevel(level)
# -----------------------------------

from common.loader import load_settings
from common.model_access import Model

# Initialize settings
settings = load_settings()

def score_answer(index: int,
                 model: Model,
                 question: str,
                 reference: str,
                 user_answer: str) -> float:
    """
    Evaluate how consistent user_answer is with reference using model.
    Returns score as float.
    """
    # Get prompts from settings with fallback to config
    scoring_prompts = getattr(settings, 'scoring_prompts', {})
    main_prompt = (scoring_prompts.get('main_mcq_prompt', None)
                  if isinstance(scoring_prompts, dict)
                  else config.score_main_prompt)
    main_system = (scoring_prompts.get('main_mcq_system', None)
                  if isinstance(scoring_prompts, dict)
                  else config.score_main_system)
    fallback_prompt_template = (scoring_prompts.get('fallback_mcq_prompt', None)
                              if isinstance(scoring_prompts, dict)
                              else config.score_fallback_prompt)
    fallback_system = (scoring_prompts.get('fallback_mcq_system', None)
                      if isinstance(scoring_prompts, dict)
                      else config.score_fallback_system)

    # Use the prompts that were found
    main_prompt = main_prompt or config.score_main_prompt
    main_system = main_system or config.score_main_system
    fallback_prompt_template = fallback_prompt_template or config.score_fallback_prompt
    fallback_system = fallback_system or config.score_fallback_system

    eval_prompt = main_prompt.format(
        question=question,
        reference_answer=reference,
        user_answer=user_answer
    )
    response = model.run(
        user_prompt=eval_prompt,
        system_prompt=main_system,
        temperature=0.0
    )
    try:
        return float(response)
    except ValueError:
        # fallback
        fallback_prompt = fallback_prompt_template.format(user_answer=user_answer)
        try:
            resp = model.run(
                user_prompt=fallback_prompt,
                system_prompt=fallback_system,
                temperature=0.0
            )
            return float(resp)
        except Exception:
            config.logger.info(f"Failed fallback scoring for index {index}, returning 0.0")
            return 0.0


def process_qa_pair(index: int,
                    qa_pair: dict,
                    modelB: Model,
                    modelA_name: str,
                    modelB_name: str) -> Optional[Tuple[int, dict, float]]:
    """
    Process one QA pair: compute score with modelB.
    Returns (index, result_dict, eval_time) or None if invalid.
    """
    question = qa_pair.get("question")
    reference = qa_pair.get("reference")
    model_answer = qa_pair.get("model")
    gen_time = qa_pair.get("gen_time")
    file = qa_pair.get("file")
    filenum = qa_pair.get("filenum")
    chunknum = qa_pair.get("chunknum")
    if not (question and reference and model_answer):
        config.logger.error(f"Skipping invalid QA at index {index}")
        return None
    start = time.time()
    score = score_answer(index, modelB, question, reference, model_answer)
    eval_time = time.time() - start
    result = {
        'modelA': modelA_name,
        'modelB': modelB_name,
        'index': index,
        'question': question,
        'reference': reference,
        'model': model_answer,
        'score': score,
        'gen_time': gen_time,
        'eval_time': f"{eval_time:.4f}",
        'file': file,
        'filenum': filenum,
        'chunknum': chunknum
    }
    return (index, result, eval_time)


def score_answers_file(
    modelA_name: str,
    modelB_name: str,
    output_dir: str,
    parallel: int = None,  # Default will come from settings/config
    force: bool = False,
    cache_dir: Optional[str] = None,
    quiet: bool = False,
    verbose: bool = False
) -> str:
    """
    Main API to score answers: reads answers JSONL, scores them, writes scores JSONL.
    Returns path to the scores file.
    """
    # configure logging & progress
    if verbose:
        config.logger.setLevel(logging.INFO)
        use_bar = False
    elif quiet:
        config.logger.setLevel(logging.CRITICAL)
        use_bar = False
    else:
        config.logger.setLevel(logging.WARNING)
        use_bar = True

    # Get default parallel workers from settings with fallback to config
    if parallel is None:
        parallel = settings.quality.defaultThreads if hasattr(settings, 'quality') else 4
    # HF cache
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        config.logger.info(f"Using HF cache dir: {cache_dir}")
    # prepare files
    answer_file = os.path.join(output_dir,
        f'answers_{modelA_name.replace("/","+")}.jsonl')
    if not os.path.exists(answer_file):
        #config.logger.error(f"Missing answers file: {answer_file}")
        #config.initiate_shutdown("No answers to score.")
        config.initiate_shutdown(f"{answer_file} missing. No answers to score. Exiting.")
    score_file = os.path.join(
        output_dir,
        f'scores_{modelA_name.replace("/","+")}={modelB_name.replace("/","+")}.jsonl'
    )
    if os.path.exists(score_file) and not force:
        config.logger.info(f"Skipping existing: {score_file}")
        return score_file
    # load data
    with open(answer_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    total = len(data)
    config.logger.info(f"Scoring {total} QA pairs using {modelB_name}")
    pbar = tqdm(total=total, desc="Scoring", unit="item") if use_bar else config.NoOpTqdm(total=total)
    # exec
    modelB = Model(modelB_name)
    modelB.details()
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(score_file):
        os.remove(score_file)
    buffer: List[dict] = []
    scores: List[float] = []
    total_eval = 0.0
    count = 0
    with open(score_file, 'a', encoding='utf-8') as out_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as exe:
            futures = [exe.submit(process_qa_pair, idx, qa, modelB, modelA_name, modelB_name)
                       for idx, qa in enumerate(data, start=1)]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                count += 1
                if res:
                    _, item, et = res
                    buffer.append(item)
                    scores.append(item['score'])
                    total_eval += et
                # Get save interval from settings with fallback to config
                save_interval = settings.quality.save_interval if hasattr(settings, 'quality') else config.saveInterval
                if len(buffer) >= save_interval:
                    for it in buffer:
                        out_f.write(json.dumps(it, ensure_ascii=False) + "\n")
                    out_f.flush()
                    buffer.clear()
                pbar.update(1)
        # flush remaining
        for it in buffer:
            out_f.write(json.dumps(it, ensure_ascii=False) + "\n")
        out_f.flush()
    pbar.close()
    # summary logs
    if scores:
        mean = statistics.mean(scores)
        var = statistics.pvariance(scores)
        std = statistics.stdev(scores) if len(scores)>1 else 0.0
        config.logger.info(
            f"Completed scoring. mean={mean:.2f}, var={var:.2f}, std={std:.2f}" )
    return score_file


def main():
    # Get defaults from settings with fallback to config
    default_model_a = settings.workflow.extraction if hasattr(settings, 'workflow') else config.model['name']
    default_model_b = settings.workflow.target if hasattr(settings, 'workflow') else config.model_b['name']
    default_threads = settings.quality.defaultThreads if hasattr(settings, 'quality') else 4
    results_dir = settings.directories.results if hasattr(settings, 'directories') else config.results_dir

    parser = argparse.ArgumentParser(
        description='Score answers: LLM B rates answers from LLM A'
    )
    parser.add_argument('-a','--modelA_name', default=default_model_a)
    parser.add_argument('-b','--modelB_name', default=default_model_b)
    parser.add_argument('-o','--output', default=results_dir)
    parser.add_argument('-f','--force', action='store_true')
    parser.add_argument('-c','--cache-dir', default=os.getenv('HF_HOME'))
    parser.add_argument('-q','--quiet', action='store_true')
    parser.add_argument('-v','--verbose', action='store_true')
    parser.add_argument('-p','--parallel', type=int, default=default_threads)
    args = parser.parse_args()
    out = score_answers_file(
        modelA_name=args.modelA_name,
        modelB_name=args.modelB_name,
        output_dir=args.output,
        parallel=args.parallel,
        force=args.force,
        cache_dir=args.cache_dir,
        quiet=args.quiet,
        verbose=args.verbose
    )
    print(out)

if __name__ == '__main__':
    main()

