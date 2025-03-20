#!/usr/bin/env python

import os
import requests
import glob
import subprocess

from config import mcq_dir, results_dir
from inference_auth_token import get_access_token
alcf_access_token = get_access_token()
from alcf_inference_utilities import get_names_of_alcf_chat_models, get_alcf_inference_service_model_queues
alcf_chat_models = get_names_of_alcf_chat_models(alcf_access_token)


def extract_model_a_from_scores_file_name(folder, file):
    part1 = file.split(f'{folder}/scores_')[1]
    part2 = part1.split('=')[0]
    return part2.replace('+', '/')


def extract_model_a_from_answers_file_name(folder, file):
    part1 = file.split(f'{folder}/answers_')[1]
    part2 = part1.split('=')[0]
    return part2.replace('+', '/')


def extract_model_b_from_scores_file_name(folder, file):
    part1 = file.split(f'{folder}/scores_')[1]
    part2 = part1.split('=')[1]
    part3 = part2.split('.json')[0]
    return part3.replace('+', '/')


def generate_scores_file_name(folder, model_a, model_b):
    return f'{folder}/scores_{model_a.replace("/","+")}:{model_b.replace("/","+")}.json'


"""
We want to run potentially many different modelAs and evaluate with many different modelBs.

If a modelA has already been run once and scored with one modelB, it can be rescored with another.

"""
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Program to run LLMs to generate and score answers to MCQs')
    parser.add_argument('-i','--inputs', help='MCQ file (default: from config.yml)', default=mcq_dir)
    parser.add_argument('-o','--outputdir', help='Directory to look for run results (default: from config.yml)', default=results_dir)
    parser.add_argument('-s','--silent', help='Just show things to run', action='store_true')
    parser.add_argument('-x','--execute', help='Execute commands', action='store_true')
    parser.add_argument('-m','--more', help='Also look at non-running/queued models', action='store_true')
    parser.add_argument('-c', "--cache-dir", type=str, default=os.getenv("HF_HOME"), help="Custom cache directory for Hugging Face")
    args = parser.parse_args()

    # Set HF_HOME if using custom cache directory
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        print(f"Using Hugging Face cache directory: {args.cache_dir}")

    # Folder containing the output files
    inputs = args.inputs
    folder = args.outputdir
    silent = args.silent
    execute= args.execute
    other  = args.more 

    running_model_list, queued_model_list = get_alcf_inference_service_model_queues(alcf_access_token)
    running_model_list = [model for model in running_model_list if model in alcf_chat_models]
    running_model_list = [model for model in running_model_list if 'batch' not in model]
    running_model_list = [model for model in running_model_list if 'auroragpt-0.1-chkpt' not in model]
    running_model_list = [model for model in running_model_list if model != 'N/A']
    running_model_list = ['alcf:'+model for model in running_model_list]
    queued_model_list = ['alcf:'+model for model in queued_model_list]
    running_model_list += ['pb:argonne-private/AuroraGPT-Tulu3-SFT-0125', 'pb:argonne-private/AuroraGPT-IT-v4-0125', 'openai:gpt-4o']

    # List available generated answers
    answers_files = [file.replace('.json', '') for file in glob.glob(f'{folder}/answers_*')]
    print('ANSWER_FILES:', answers_files)
    if not silent:
        print(f'\nReviewing answers and scores files in {folder}, looking for MCQs not answered and/or scored with running models.')

    models_scored = {}

    if not silent:
        print(f'====== Models running, queued, available at ALCF inference service ====')
        print(f'Running models: {running_model_list}')
        print(f'Queued models : {queued_model_list}')
        other_models = [model for model in alcf_chat_models if model not in running_model_list and model not in queued_model_list]
        print(f'Other models : {other_models}')

        # List for each set of answers which models have reviewed it
        print(f'\n====== Answers and scores obtained to date for {folder} ========')
        for file in answers_files:
            model_a = file.split("answers_")[1].replace('+','/')
            print(f'{model_a}')
            score_files = glob.glob(f'{folder}/scores_{model_a.replace("/","+")}:*')
            m_list = []
            for score_file in score_files:
                print('SF', score_file)
                f = score_file.split(f'{folder}/scores_{model_a.replace("/","+")}=')[1]
                model_b = f.split("_")[0].replace('+','/').replace('.json','')
                print(f'\t{model_b}')
                m_list.append(model_b)
            models_scored[model_a] = m_list


    no_answer_list = [f'python generate_answers.py -o {folder} -i {inputs} -m {model_a}' for model_a in running_model_list if not os.path.isfile(f'{folder}/answers_{model_a.replace("/","+")}.json')]

    # List for each possible reviewer (i.e., a running model) which answers it has not reviewed
    no_score_list = []
    for model_b in running_model_list:
        for filename in answers_files:
            model_a = extract_model_a_from_answers_file_name(folder, filename)
            if not os.path.isfile(generate_scores_file_name(folder, model_a, model_b)):
                score_filename = generate_scores_file_name(folder, model_a, model_b)
                command = f'python score_answers.py -o {folder} -a {model_a} -b {model_b}'
                no_score_list.append(command)

    if not silent and (no_answer_list != [] or no_score_list != []):
        print('\n======= Commands that may be executed based on currently running models ==================')
        if no_answer_list != []:
            print('a) To generate answers')
            for command in no_answer_list:
                print(f'    {command}')
        else:
            print('a) To generate answers: None')
        if no_score_list != []:
            print('b) To generate scores')
            for command in no_score_list:
                print(f'    {command}')
        else:
            print('b) To generate scores: None')

    # List running models that have not generated answers
    if no_answer_list != [] and not silent and execute:
        print('\n====== Generating answers with running models ============================================')
        for command in no_answer_list:
            print(f'    Executing {command}')
            try:
               subprocess.run(command, shell=True)
            except OSError as e:
                print(f'    Error {e}')
                return -1

    if not silent and no_score_list!=[] and execute:
        print('\n====== Generating scores for answers that can be provided by a running model =============')
        for command in no_score_list:
            print(f'    Executing {command}')
            try:
                subprocess.run(command, shell=True)
            except OSError as e:
                print(f'    Error {e}')
                return -1

    if other:
       print('\n====== Non-running/queued models ======')
       for model_a in other_models:
           if not os.path.isfile(f'{folder}/answers_{model_a.replace("/","+")}.json'):
               print(f'\npython generate_answers.py -o {folder} -i {folder}.json -m {model_a}')

if __name__ == "__main__":
    main()
