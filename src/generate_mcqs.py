#!/usr/bin/env python

# parallel_generate_mcqs.py

import os
import sys
import json
import re
import time  # For timing
from openai import OpenAI
import spacy
import argparse
import config
import logging
from tqdm import tqdm  # For progress bars
import concurrent.futures
import threading

from model_access import Model

##############################################################################
# Global constants
##############################################################################
CHUNK_SIZE = config.chunkSize

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")


def human_readable_time(seconds: float) -> str:
    """
    Convert time in seconds into a more human-friendly format.
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.2f} hours"
    else:
        days = seconds / 86400
        return f"{days:.2f} days"


def approximate_total_chunks(input_dir, output_dir, chunk_size=CHUNK_SIZE):
    total_chunks = 0
    for f in os.listdir(input_dir):
        if not f.lower().endswith((".json", ".jsonl")):
            continue
        output_file = os.path.join(output_dir, f"processed_{f}")
        if os.path.exists(output_file):
            continue
        path = os.path.join(input_dir, f)
        try:
            with open(path, 'r', encoding='utf-8') as file:
                if f.lower().endswith(".json"):
                    json_str = file.read()
                    lines = [json_str]
                else:
                    lines = file.readlines()
                
                for line in lines:
                    try:
                        record = json.loads(line.strip())
                        text = record.get('text')
                        if text:
                            chunks = split_text_into_chunks(text, chunk_size)
                            total_chunks += len(chunks)
                    except json.JSONDecodeError as e:
                        config.logger.info(f"JSON decode error in file {f}: {e}")
                        continue
        except Exception as e:
            config.logger.error(f"Failed to read file {path}: {e}")
            continue
    return total_chunks


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """
    Split the text into chunks of ~chunk_size words, respecting sentence
    boundaries using spaCy for sentence segmentation.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_length + word_count > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = word_count
        else:
            current_chunk.append(sentence)
            current_length += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Helper function to update shared counters and check ratio
def update_shared_counters(is_success, shared_counters, counter_lock):
    with counter_lock:
        if is_success:
            shared_counters["success"] += 1
        else:
            shared_counters["failure"] += 1
        total = shared_counters["success"] + shared_counters["failure"]
        if total > 8:
            ratio = shared_counters["success"] / total
            if ratio < 0.5:
                config.logger.error(
                    f"Success rate <50% ({shared_counters['success']}/{total}).\n"
                    f"       To bail, hit ^C multiple times to abort all threads.\n"
                    f"       Run with -v (--verbose) to investigate."
                )
                sys.exit(1)


def process_chunk(model, filename, file_path, linenum, chunknum, chunk,
                  pbar_total, pbar_success, shared_counters, counter_lock):
    """
    Process a single chunk. This function performs the three steps:
      1. Summarize & expand the chunk.
      2. Generate the multiple-choice question.
      3. Verify and score the result.

    Returns a tuple containing:
      (filename, linenum, chunknum, result_dict or None, success_flag)
    """
    chunk_success = False
    qa_pair = None

   # New code to check if shutdown has been requested
    if config.bail_out:
        config.logger.info(f"Graceful shutdown: Skipping chunk {chunknum} in file {filename}.")
        return (filename, linenum, chunknum, None, False)


    # Log the start of processing this chunk
    config.logger.info(f"Processing chunk {chunknum} in file {filename}.")

    # Step 1: Summarize & expand the chunk
    try:
        formatted_user_message = config.user_message.format(chunk=chunk)
        step1_output = model.run(user_prompt=formatted_user_message,
                                 system_prompt=config.system_message)
        augmented_chunk = step1_output
        if "augmented_chunk:" in str(step1_output).lower():
            augmented_chunk = re.split(
                r'augmented_chunk\s*:\s*',
                step1_output,
                flags=re.IGNORECASE,
                maxsplit=1
            )[-1].strip()
    except Exception as e:
        config.logger.warning(f"Error summarizing chunk {chunknum} in file {filename}: {e}")
        pbar_total.update(1)
        update_shared_counters(False, shared_counters, counter_lock)
        return (filename, linenum, chunknum, None, False)

    # Step 2: Generate the multiple-choice question
    try:
        formatted_user_message_2 = config.user_message_2.format(augmented_chunk=augmented_chunk)
        generated_question = model.run(user_prompt=formatted_user_message_2,
                                       system_prompt=config.system_message_2)
    except Exception as e:
        config.logger.warning(f"Error generating question for chunk {chunknum} in file {filename}: {e}")
        pbar_total.update(1)
        update_shared_counters(False, shared_counters, counter_lock)
        return (filename, linenum, chunknum, None, False)

    # Step 3: Verify the question and score it
    try:
        formatted_user_message_3 = config.user_message_3.format(
            augmented_chunk=augmented_chunk,
            generated_question=generated_question
        )
        step3_output = model.run(
            user_prompt=formatted_user_message_3,
            system_prompt=config.system_message_3
        )
        if step3_output is None:
            raise ValueError("model.run() returned None for step3_output.")

        # Clean and parse the JSON output
        step3_output = step3_output.replace("```json", "").replace("```", "")
        step3_output = step3_output.replace('\\"', "XXXABCXXX")
        step3_output = step3_output.replace("\\", "\\\\")
        step3_output = step3_output.replace("XXXABCXXX", '\\"')

        parsed_json = json.loads(step3_output)
        if isinstance(parsed_json, str):
            parsed_json = json.loads(parsed_json)
        if not isinstance(parsed_json, dict):
            raise ValueError(f"Expected a JSON object but got: {parsed_json}")

        model_answer = str(parsed_json.get("answer", "")).strip()
        model_score = parsed_json.get("score", 0)
        pbar_total.set_postfix_str(f"Score: {model_score}")

        if isinstance(model_score, int) and model_score > config.minScore:
            config.logger.info(f"MCQ generated for chunk {chunknum} in file {filename}, score {model_score} > {config.minScore}.")
            qa_pair = {
                "file": filename,
                "path": file_path,
                "line": linenum,
                "chunk": chunknum,
                "model": model.model_name,
                "question": generated_question,
                "answer": model_answer,
                "text": augmented_chunk
            }
            chunk_success = True
        else:
            config.logger.info(f"MCQ generation failed for chunk {chunknum} in file {filename}.")
    except json.JSONDecodeError:
        config.logger.info(f"Chunk JSON parsing failed for chunk {chunknum} in file {filename}. Trying to fix.")
        fix_prompt = f"""
        Convert the following text strictly into valid JSON with three key/value
        pairs: question, answer, score. Nothing else, no additional text.

        TEXT TO FIX:
        {step3_output}
        """
        try:
            fixed_json_output = model.run(
                system_prompt="You are a strict JSON converter.",
                user_prompt=fix_prompt
            )
            parsed_json = json.loads(fixed_json_output)
            if isinstance(parsed_json, str):
                parsed_json = json.loads(parsed_json)
        except json.JSONDecodeError as e:
            config.logger.info(f"Chunk fix failed for chunk {chunknum} in file {filename}: {e}")
            pbar_total.update(1)
            update_shared_counters(False, shared_counters, counter_lock)
            return (filename, linenum, chunknum, None, False)

        model_answer = parsed_json.get("answer", "").strip()
        model_score = parsed_json.get("score", 0)
        pbar_total.set_postfix_str(f"Score: {model_score}")
        if isinstance(model_score, int) and model_score > config.minScore:
            qa_pair = {
                "file": filename,
                "path": file_path,
                "line": linenum,
                "chunk": chunknum,
                "model": model.model_name,
                "question": generated_question,
                "answer": model_answer,
                "text": augmented_chunk
            }
            chunk_success = True
        else:
            config.logger.info(f"Chunk fix unsuccessful for chunk {chunknum} in file {filename}.")
    except Exception as e:
        config.logger.info(f"Error verifying chunk {chunknum} in file {filename}: {e}")
        pbar_total.update(1)
        update_shared_counters(False, shared_counters, counter_lock)
        return (filename, linenum, chunknum, None, False)

    # Finally, update progress bar and shared counters for this chunk
    pbar_total.update(1)
    update_shared_counters(chunk_success, shared_counters, counter_lock)
    if chunk_success:
        pbar_success.update(1)
    return (filename, linenum, chunknum, qa_pair, chunk_success)


def process_directory(model, input_dir: str, output_dir: str = "output_files",
                      use_progress_bar: bool = True, parallel_workers: int = 4):
    """
    Process all JSON/JSONL files by scheduling each chunk as a separate task.
    Once all tasks are done, group results by file and write output files.
    Also creates shared counters (with a lock) that are updated on each chunk.
    """
    json_files  = [f for f in os.listdir(input_dir) if f.lower().endswith(".json")]
    jsonl_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jsonl")]
    all_files = json_files + jsonl_files
    total_files = len(all_files)

    if total_files == 0:
        config.logger.warning(f"No JSON files found in {input_dir}.")
        return

    overall_start_time = time.time()

    if json_files:
        approximate_chunk_count = approximate_total_chunks(input_dir, output_dir, chunk_size=CHUNK_SIZE)

        config.logger.info(f"\nTotal JSON files: {total_files}, ~{int(0.8 * approximate_chunk_count)}-{approximate_chunk_count} chunks\n")
    else:
        approximate_chunk_count = sum(1 for _ in open(os.path.join(input_dir, jsonl_files[0]), 'r', encoding='utf-8'))

    # Create shared counters and lock for chunk-level progress
    counter_lock = threading.Lock()
    shared_counters = {"success": 0, "failure": 0}

    # Dictionary to collect results by filename
    file_results = {}

    if use_progress_bar:
        pbar_total = tqdm(total=approximate_chunk_count, desc=" Processed", position=0, unit="chunk")
        pbar_success = tqdm(total=approximate_chunk_count, desc="Successful", position=1, unit="chunk")
    else:
        pbar_total = config.NoOpTqdm()
        pbar_success = config.NoOpTqdm()

    # We'll collect all chunk-level tasks in a list
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        for filename in all_files:
            file_path = os.path.join(input_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    if filename.lower().endswith(".json"):
                        json_str = file.read()
                        lines = [json_str]
                    else:
                        lines = file.readlines()
            except Exception as e:
                config.logger.error(f"Failed to read file {filename}: {e}")
                continue

            # Process each line (record) in the file
            for linenum, line in enumerate(lines, start=1):
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    config.logger.info(f"JSON decode error in file {filename} line {linenum}: {e}")
                    continue

                text = record.get('text')
                rec_path = record.get('path', file_path)
                if not text:
                    continue
                chunks = split_text_into_chunks(text, CHUNK_SIZE)
                # For each chunk in this record, submit a task
                for chunknum, chunk in enumerate(chunks, start=1):
                    if config.bail_out:
                        config.logger.info("Shutdown in progress: stopping submission of new tasks (chunk loop).")
                        break

                    future = executor.submit(process_chunk, model, filename, rec_path,
                                             linenum, chunknum, chunk,
                                             pbar_total, pbar_success,
                                             shared_counters, counter_lock)
                    futures.append(future)

        # Now, collect the results from all futures.
        for future in concurrent.futures.as_completed(futures):
            try:
                fname, linenum, chunknum, qa_pair, success = future.result()
                if qa_pair is not None:
                    file_results.setdefault(fname, []).append(qa_pair)
            except Exception as e:
                config.logger.error(f"Error processing a chunk: {e}")

    if use_progress_bar:
        remaining = pbar_total.total - pbar_total.n
        if remaining > 0:
            pbar_total.update(remaining)
        pbar_total.close()
        pbar_success.close()

# Write out results grouped by file
    os.makedirs(output_dir, exist_ok=True)
    for fname, qa_pairs in file_results.items():
        out_file = os.path.join(output_dir, f'processed_{fname}')
        config.logger.info(f"Writing output for file {fname} with {len(qa_pairs)} MCQs to {out_file}")
        try:
            with open(out_file, 'w', encoding='utf-8') as out_f:
                json.dump(qa_pairs, out_f, ensure_ascii=False, indent=2)
        except Exception as e:
            config.logger.error(f"Failed to write output file {out_file}: {e}")

    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    config.logger.info(
        f"Processed {total_files} files in {human_readable_time(total_time)}.\n"
        f"Shared counters: {shared_counters}\n"
        f"Prompt/answer pairs saved to {output_dir}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Program to generate MCQs from JSONL or JSON files')
    parser.add_argument('-i', '--input', help='Directory containing input JSON/JSONL files', default=config.json_dir)
    parser.add_argument('-o', '--output', help='Output directory for MCQs', default=config.mcq_dir)
    parser.add_argument('-m', '--model', help='Model to use to generate MCQs', default=config.defaultModel)
    parser.add_argument('-q', '--quiet', action='store_true', help='No progress bar or messages')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('-p', '--parallel', type=int, default=4, help='Number of parallel threads (default: 4)')

    args = parser.parse_args()

    use_progress_bar = config.configure_verbosity(args)

    input_directory = args.input
    output_json = args.output

    model_name = args.model
    model = Model(model_name)
    model.details()

    try:
        process_directory(model, input_directory, output_json,
                          use_progress_bar=use_progress_bar,
                          parallel_workers=args.parallel)
    except KeyboardInterrupt:
        print("EXIT: Execution interrupted by user")
        sys.exit(0)

