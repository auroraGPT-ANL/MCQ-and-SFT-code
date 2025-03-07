#!/usr/bin/env python

import sys
import json
import os
import statistics
import requests
#import openai
#from openai import OpenAI, APITimeoutError
import time
import glob
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

    # print(f'\n=========== SCORE ANSWER {index} ===================\n\n-------- QUESTION -----\n{question}\n------ USER ANSWER\n{user_answer}\n ====================\n\n')

    # We ask the model to strictly return a single number from 1 to 10
    # indicating how well the user_answer matches the reference_answer.
    eval_prompt = f"""
You are a strict grader. 

Question: {question}
Reference Answer: {reference_answer}
User's Answer: {user_answer}

On a scale of 1 to 10 (10 = exactly matches the reference answer, 
1 = completely incorrect), provide ONLY the numeric score that reflects
how well the User's Answer matches the Reference Answer. Provide just a number. No extra text. No explanation. No formatting.
"""

    response = model.run( user_prompt=eval_prompt,
                          system_prompt="You are a strict grader. Respond with only the number.",
                          temperature  = 0.0 )

### We may want to return 0.0 above to be consistent with below

    # Extract the numeric score from the assistant's response
    try:
        score = float(response)
    except ValueError:
        score = try_again_to_extract_score(user_answer)
        # If the model didn't return a pure number, attempt a fallback or default to 0
        #print(f'Score of 0 for bad response++++\n{response}\n+++++++++\n')
        #print(f'\n\n----------\n{eval_prompt}\n\n==================')
        #score = 0.0
    
    return score


def try_again_to_extract_score(user_answer):
    try:
        prompt = f"Extract a final answer from this user response. Sometimes this appears at the end after the words '**Final Answer**', enclosed in the characters '\\[ \\boxed{' and '} \\]'. For example, '\\[ \\boxed{3} \\]' for the answer '3'.\n\nHere is the user response:\n{user_answer}"
        response = model.run( user_prompt=prompt,
                             system_prompt="Extract a single number",
                             temperature  = 0.0 )
        score = float(response)
    except:
        config.logger.info(f'Score of 0 for bad response++++\n{user_answer}\n+++++++++\n')
        score = 0.0

    return score



def main():
    parser = argparse.ArgumentParser(description='Program to use LLM B to rate answers provided previously by LLM A') 
    parser.add_argument('-a','--modelA_name', help='modelA name', required=True)
    parser.add_argument('-b','--modelB_name', help='modelB name', required=True)
    parser.add_argument('-o','--output', help='Output directory', required=True)
    parser.add_argument('-f','--force',  help='Process even if score file exists',
                        action="store_true")
    parser.add_argument('-c', "--cache-dir", type=str,
                        default=os.getenv("HF_HOME"),
                        help="Custom cache directory for Hugging Face")
    parser.add_argument('-q','--quiet',   action='store_true',   
                        help='No progress bar or messages')
    parser.add_argument('-v','--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Decide logging level and whether to show a progress bar
    if args.verbose:
        config.logger.setLevel(logging.INFO)
        use_progress_bar = False
    elif args.quiet:
        config.logger.setLevel(logging.CRITICAL)
        use_progress_bar = False
    else:  # default case
        config.logger.setLevel(logging.WARNING)
        use_progress_bar = True


    # Set HF_HOME if using custom cache directory
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        config.logger.info(f"Using Hugging Face cache directory: {args.cache_dir}")

    output_dir   = args.output

    modelA_name = args.modelA_name
    modelB_name = args.modelB_name
    modelB      = Model(modelB_name)

    # Load previously generated answers from modelA
    answer_file = output_dir+'/answers_'+modelA_name.replace('/', '+')+'.json'
    config.logger.info(f'Looking for {answer_file}')
    if not os.path.exists(answer_file):
        config.logger.error(f'No answers file for {modelA_name}')
        exit(1)

    score_file = f'{output_dir}/scores_{modelA_name.replace("/","+")}={modelB_name.replace("/","+")}.json'
    if os.path.exists(score_file) and not args.force:
        config.logger.error(f"Score file already exists: {score_file}")
        exit(1)

    out_f = open(score_file, 'w', encoding='utf-8') 

    # Load question-answer pairs
    with open(answer_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores   = []
    qa_pairs = []

    # Create progress bar if not in --quiet or --verbose mode.
    if use_progress_bar:
        pbar = tqdm(total=len(data), desc="Processing", unit="item")
    else:
        pbar = config.NoOpTqdm(total=len(data))

    start_time = time.time()
    total_time = 0
    eval_answer_total_time = 0

    config.logger.info(f'Processing {len(data)} QA pairs')
    for (qa_pair, index) in zip(data, range(1, len(data) + 1)):
        question         = qa_pair.get("question", "")
        reference_answer = qa_pair.get("reference", "")
        model_answer     = qa_pair.get("model", "")
        gen_time         = qa_pair.get("gen_time", "")
        file             = qa_pair.get("file", "")
        filenum          = qa_pair.get("filenum", "")
        chunknum         = qa_pair.get("chunknum", "")

        if not question or not reference_answer or not model_answer:
            config.logger.error('Bad item:')
            config.logger.error('Question ', question)
            config.logger.error('Reference', reference_answer)
            config.logger.error('Model    ', model_answer)
            exit(1)
            continue  # skip malformed items

        # Use model to evaluate/grade the generated answer in file
        # against the reference answer
        eval_answer_start_time = time.time()
        score = score_answer(index, modelB, question, reference_answer, model_answer)
        eval_answer_time = time.time() - eval_answer_start_time
        eval_answer_total_time += eval_answer_time
        if score != None:
            scores.append(score)
            qa_pairs.append({'modelA': modelA_name, 'modelB': modelB_name, 'index': index, 'question': question, 'reference':reference_answer, 'model':model_answer, 'score':score, 'gen_time': gen_time, 'eval_time': f'{eval_answer_time:.4f}', 'file':file, 'filenum':filenum, 'chunknum':chunknum})

        total_time += time.time() - start_time
        start_time = time.time()
        if index%10==0:
            avg_time = total_time / index  # Average time per item so far
            avg_eval_time = eval_answer_total_time / index  # Average time per item so far
            config.logger.info(f'{index} ({avg_time:.2f})', end =' ', flush=True) 
        pbar.update(1)
    
    config.logger.info()

    json.dump(qa_pairs, out_f, ensure_ascii=False, indent=2)

    pbar.close()

    if scores:
        mean_score = statistics.mean(scores)
        variance_score = statistics.pvariance(scores)  # population variance
    else:
        config.logger.warning("No valid QA pairs found or no scores computed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        config.logger.warning("EXIT: Execution interrupted by user")
        sys.exit(0)

