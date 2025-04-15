#!/usr/bin/env python

import os
import re
import subprocess
import requests
import openai
import json

from inference_auth_token import get_access_token
alcf_access_token = get_access_token()

# Define the URL and headers for ALCF Inference Service list-endpoints
url = "https://data-portal-dev.cels.anl.gov/resource_server/list-endpoints"
headers = {
    "Authorization": f"Bearer {alcf_access_token}"
}

# Make the GET request
response = requests.get(url, headers=headers)

# Check the response
if response.status_code == 200:
    status = response.json()
else:
    print("Error:", response.status_code, response.text)
    exit(1)

alcf_chat_models = status['clusters']['sophia']['frameworks']['vllm']['models']

def parse_existing_ranges(folder, model_a, model_b):
    # Collect existing ranges from filenames
    existing_ranges = []
    if not os.path.exists(folder): os.makedirs(folder)
    for filename in os.listdir(folder):
        parts = filename.split('_')
        # Replace '/' with '+' in model names for filename compatibility
        if parts[1] != f'{model_a.replace("/","+")}:{model_b.replace("/","+")}':
            continue
        
        start = int(parts[2])
        end   = int(parts[3].split('.')[0])
        existing_ranges.append((start, end))

    return sorted(existing_ranges)

def find_missing_ranges(existing_ranges, total_range):
    missing_ranges = []
    start = total_range[0]

    for s, e in existing_ranges:
        if start < s:
            missing_ranges.append((start, s))
        start = max(start, e)

    if start < total_range[1]:
        missing_ranges.append((start, total_range[1]))

    return missing_ranges


def generate_commands(inputs, folder, model_a, model_b, total_range, batch_size):
    # Parse existing ranges from filenames
    existing_ranges = parse_existing_ranges(folder, model_a, model_b)

    total_already_done = sum(end - start for start, end in existing_ranges)
    #print(f'{total_already_done} of {total_range[1]} done so far')

    # Find missing ranges
    missing_ranges = find_missing_ranges(existing_ranges, total_range)

    # Generate commands for missing ranges, batching by batch_size
    commands = []
    for start, end in missing_ranges:
        current = start
        if current >= total_range[1]:
             break
        while current < end and current < total_range[1]:
            batch_end = min((current + batch_size)-(current + batch_size)%batch_size, end)
            commands.append(
                f"python generate_and_grade_answers.py -i {inputs} -o {folder} -a '{model_a}' -b '{model_b}' -c -q -s {current} -e {batch_end}"
            )
            current = batch_end

    return commands


def run_requested(inputs, folder, model_a, model_b, total_range, batch_size, execute): 
    print(f"Commands to run for models '{model_a}' and '{model_b}' for range {total_range}:")

    # Generate commands
    commands = generate_commands(inputs, folder, model_a, model_b, total_range, batch_size)
    if commands == []:
        print("    No commands to run")

    # Print and optionally execute the commands
    for command in commands:
        if execute:
            print(f'    Executing {command}')
            try:
                subprocess.run(command, shell=True)
            except OSError as e:
                print(f'    Error {e}')
                return -1
        else:
            print(f'    {command}')


def get_models():
    # Define the URL and headers
    url = "https://data-portal-dev.cels.anl.gov/resource_server/sophia/jobs"
    headers = { "Authorization": f"Bearer {alcf_access_token}" }

    # Make the GET request
    response = requests.get(url, headers=headers)

    # Check the response
    if response.status_code == 200:
        status = response.json()
    else:
        print("Error:", response.status_code, response.text)
        exit(1)

    running_models = status['running']
    running_model_list = []
    for model in running_models:
        running_model_list += model['Models Served'].split(',')
        if model['Model Status'] != 'running':
            print(f'SURPRISE: Expected *running* but got {model["Model Status"]}')

    queued_models = status['queued']
    queued_model_list = []
    for model in queued_models:
        queued_model_list += model['Models Served'].split(',')
        if model['Model Status'] != 'starting':
            print(f'SURPRISE: Expected *starting* but got {model["Model Status"]}')

    return(running_model_list, queued_model_list)


"""
We want to run potentially many different modelAs and evaluate with many different modelBs.

If a modelA has already been run once and scored with one modelB, it can be rescored with another.



"""
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Program to run LLM B to rate answers provided by LLM A')
    parser.add_argument('-a','--modelA', help='modelA', required=True)
    parser.add_argument('-b','--modelB', help='modelB', default='gpt-4o')
    parser.add_argument('-o','--outputdir', help='Directory to look for run results', required=True)
    parser.add_argument('-i','--inputfile', help='File to look for inputs', required=True)
    parser.add_argument('-x','--execute', help='Run program', action="store_true")
    parser.add_argument('-q','--queued', help='Process queued models', action="store_true")
    parser.add_argument('-m','--max', help='Max to process', default=0)
    parser.add_argument('-s','--start', help='Request to non-running models', action="store_true")
    args = parser.parse_args()
    execute = args.execute

    model_a = args.modelA
    model_b = args.modelB

    # Folder containing the output files
    folder = args.outputdir
    inputs = args.inputfile

    with open(inputs, "r", encoding="utf-8") as f:
        data = json.load(f)
        max_value = len(data)

    if int(args.max) > 0 and int(args.max) < max_value:
        max_value = int(args.max)

    if execute:
        print(f'Running on {max_value} elements of {inputs} with {model_b} as judge, saving results to {folder}.')
    else:
        print(f'Checking what is runnable on {max_value} elements of {inputs} with {model_b} as judge.')

    # Total range to cover
    total_range = (0, max_value)
    
    # Batch size for each run
    batch_size = 100

    running_model_list, queued_model_list = get_models()

    running_model_list = [model for model in running_model_list if 'auroragpt-0.1-chkpt' not in model]
    queued_model_list = [model for model in queued_model_list if 'auroragpt-0.1-chkpt' not in model]

    if model_a == 'all':
        print(f'Requested all. Trying {running_model_list}, currently running at ALCF\n')
        for model_a in running_model_list:
            if model_a not in alcf_chat_models:
                print(f'Skipping {model_a} as not a chat model')
                continue
            run_requested(inputs, folder, model_a, model_b, total_range, batch_size, execute)
        if args.queued:
            print(f'Also trying {queued_model_list}, currently queued at ALCF\n')
            for model_a in queued_model_list:
                if model_a not in alcf_chat_models:
                    print(f'Skipping {model_a} as not a chat model')
                    continue
                run_requested(inputs, folder, model_a, model_b, total_range, batch_size, execute)
        if args.start:
            print(f'Trying all ALCF chat models\n')
            all_model_list = alcf_chat_models
            all_model_list = [model for model in all_model_list if 'auroragpt-0.1-chkpt' not in model]
            for model_a in all_model_list:
                run_requested(inputs, folder, model_a, model_b, total_range, batch_size, execute)
    else:
        if model_a in running_model_list:
            run_requested(inputs, folder, model_a, model_b, total_range, batch_size, execute)
        elif args.queued and model_a in queued_model_list:
            print(f'Requesting {model_a}, which is currently queued')
            run_requested(inputs,folder, model_a, model_b, total_range, batch_size, execute)
        elif args.start and model_a in alcf_chat_models:
            print(f'Requesting {model_a} start')
            run_requested(inputs, folder, model_a, model_b, total_range, batch_size, execute)
        else:
            print(f'Model {model_a} is not not known')
            exit(1)

if __name__ == "__main__":
    main()

