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
from concurrent.futures import TimeoutError
import threading

from model_access import Model

##############################################################################
# Global constants
##############################################################################
CHUNK_SIZE = config.chunkSize

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")


def human_readable_time(seconds: float) -> str:
    """Convert time in seconds into a human-friendly format."""
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


def count_chunks_in_file(filepath, chunk_size=CHUNK_SIZE):
    """
    Reads a file (JSON or JSONL) and returns the total number of chunks
    contained in it using the same split_text_into_chunks function.
    """
    total_chunks = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            if filepath.lower().endswith(".json"):
                json_str = file.read()
                lines = [json_str]
            else:
                lines = file.readlines()
            for line in lines:
                try:
                    record = json.loads(line.strip())
                    text = record.get('text', '')
                    if text:
                        chunks = split_text_into_chunks(text, chunk_size)
                        total_chunks += len(chunks)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        config.logger.error(f"Failed to count chunks in file {filepath}: {e}")
    return total_chunks


def approximate_total_chunks(input_dir, output_dir, chunk_size=CHUNK_SIZE, force=False):
    """
    Estimate total chunks for files in input_dir.
    If force is False, files with an existing processed output are skipped.
    """
    total_chunks = 0
    for f in os.listdir(input_dir):
        if not f.lower().endswith((".json", ".jsonl")):
            continue
        output_file = os.path.join(output_dir, f"processed_{os.path.splitext(f)[0]}.jsonl")
        if os.path.exists(output_file) and not force:
            # Count the chunks in the already processed file
            total_chunks += count_chunks_in_file(os.path.join(input_dir, f), chunk_size)
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
            if config.shutdown_event.is_set():
                config.logger.info("Shutdown in progress; suppressing error details.")
                return 0
            else:
                config.logger.error(f"Failed to read file {path}: {e}")
                continue
    return total_chunks


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """Split text into chunks of roughly chunk_size words using spaCy for sentence segmentation."""
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


# Helper function to update shared counters.
def update_shared_counters(is_success, shared_counters, counter_lock):
    with counter_lock:
        if is_success:
            shared_counters["success"] += 1
        else:
            shared_counters["failure"] += 1
        total = shared_counters["success"] + shared_counters["failure"]
        if total > 16:
            ratio = shared_counters["success"] / total
            if ratio < 0.5:
                config.logger.error(
                    f"Success rate <50% ({shared_counters['success']}/{total}).\n"
                    f"       Run with -v (--verbose) to investigate."
                )
                config.initiate_shutdown("Initiating shutdown.")


# Helper function to robustly parse JSON output.
def robust_parse_json_output(response_text, model):
    """
    Attempts to robustly parse a JSON response expected to contain three keys:
    "answer", "score", and "comment". Cleans the text and checks for cases
    where the output is a single integer or in the form "X) Answer text".
    If parsing fails, issues a fix prompt.
    """
    cleaned = response_text.replace("```json", "").replace("```", "").strip()
    if cleaned.isdigit():
        return {"answer": "", "score": int(cleaned), "comment": ""}
    m = re.match(r"^(\d+)\)\s*(.*)", cleaned)
    if m:
        return {"answer": m.group(2).strip(), "score": int(m.group(1)), "comment": ""}
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        if isinstance(parsed, dict):
            return parsed
        else:
            raise ValueError(f"Parsed JSON is not an object: {parsed}")
    except Exception as e:
        try:
            with open("json_error_log.txt", "a", encoding="utf-8") as error_file:
                error_file.write(f"Initial JSON parsing error: {e}\nResponse: {response_text}\n\n")
                error_file.flush()
        except Exception as file_e:
            config.logger.error(f"Failed to write initial JSON error: {file_e}")
        config.logger.info(f"Initial JSON parsing failed: {e}. Attempting reformat.")
        fix_prompt = f"""
Convert the following text strictly into valid JSON with exactly three keys: "question", "answer", and "score".
Do not include any additional text or markdown.
TEXT TO FIX:
{response_text}
        """
        fixed_output = model.run(
            system_prompt="You are a strict JSON converter. Return only valid JSON.",
            user_prompt=fix_prompt
        )
        if not fixed_output.strip():
            if cleaned.isdigit():
                return {"answer": "", "score": int(cleaned), "comment": ""}
            else:
                raise ValueError("Fix prompt returned empty output.")
        try:
            parsed_fixed = json.loads(fixed_output)
            if isinstance(parsed_fixed, str):
                parsed_fixed = json.loads(parsed_fixed)
            if isinstance(parsed_fixed, dict):
                return parsed_fixed
            else:
                raise ValueError(f"Fixed parsed JSON is not an object: {parsed_fixed}")
        except Exception as fix_error:
            try:
                with open("json_error_log.txt", "a", encoding="utf-8") as error_file:
                    error_file.write(f"Fix JSON parsing error: {fix_error}\nFixed output: {fixed_output}\n\n")
                    error_file.flush()
            except Exception as file_e:
                config.logger.error(f"Failed to write fix error: {file_e}")
            raise ValueError(f"Failed to reformat JSON output. Original error: {e}; fix error: {fix_error}")


def process_chunk(model, filename, file_path, linenum, chunknum, chunk,
                  pbar_total, pbar_success, shared_counters, counter_lock):
    """
    Process a single chunk in three steps:
      1. Summarize & expand the chunk.
      2. Generate the multiple-choice question.
      3. Verify and score the result.
    Returns a tuple: (filename, linenum, chunknum, result_dict or None, success_flag)
    """
    chunk_success = False
    qa_pair = None

    if config.shutdown_event.is_set():
        #config.logger.info(f"Shutting down: Skipping chunk {chunknum} in file {filename}.")
        return (filename, linenum, chunknum, None, False)

    config.logger.info(f"Processing chunk {chunknum} in file {filename}.")

    # Step 1: Summarize & expand.
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
        if config.shutdown_event.is_set():
            config.logger.info("Shutdown in progress; suppressing error details.")
            return (filename, linenum, chunknum, None, False)
        else:
            config.logger.warning(f"Error summarizing chunk {chunknum} in file {filename}: {e}")
            pbar_total.update(1)
            update_shared_counters(False, shared_counters, counter_lock)
            return (filename, linenum, chunknum, None, False)

    # Step 2: Generate the multiple-choice question.
    if config.shutdown_event.is_set():
        #config.logger.info(f"Shutting down: Skipping step 2 in chunk {chunknum} in file {filename}.")
        return (filename, linenum, chunknum, None, False)
    try:
        formatted_user_message_2 = config.user_message_2.format(augmented_chunk=augmented_chunk)
        generated_question = model.run(user_prompt=formatted_user_message_2,
                                       system_prompt=config.system_message_2)
    except Exception as e:
        if config.shutdown_event.is_set():
            config.logger.info("Shutdown in progress; suppressing error details.")
            return (filename, linenum, chunknum, None, False)
        else:
            config.logger.warning(f"Error generating question for chunk {chunknum} in file {filename}: {e}")
            pbar_total.update(1)
            update_shared_counters(False, shared_counters, counter_lock)
            return (filename, linenum, chunknum, None, False)

    # Step 3: Verify the question and score it.
    if config.shutdown_event.is_set():
        #config.logger.info(f"Shutting down: Skipping step 3 in chunk {chunknum} in file {filename}.")
        return (filename, linenum, chunknum, None, False)
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
        step3_clean = step3_output.replace("```json", "").replace("```", "").strip()
        parsed_json = robust_parse_json_output(step3_clean, model)
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
    except Exception as e:
        config.logger.info(f"Error verifying chunk {chunknum} in file {filename}: {e}")
        pbar_total.update(1)
        update_shared_counters(False, shared_counters, counter_lock)
        return (filename, linenum, chunknum, None, False)

    pbar_total.update(1)
    update_shared_counters(chunk_success, shared_counters, counter_lock)
    if chunk_success:
        pbar_success.update(1)
    return (filename, linenum, chunknum, qa_pair, chunk_success)


def process_directory(model, input_dir: str, output_dir: str = "output_files",
                      use_progress_bar: bool = True, parallel_workers: int = 4, force=False):
    """
    Process all JSON/JSONL files by scheduling each chunk as a separate task.
    If --force is not specified and a processed file exists for a given input file,
    the file is skipped (and its chunk count is added as successful). With --force,
    the file is reprocessed and the new MCQs are appended to the existing output.
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
        approximate_chunk_count = 0
        for f in all_files:
            filepath = os.path.join(input_dir, f)
            approximate_chunk_count += count_chunks_in_file(filepath, CHUNK_SIZE)
        config.logger.info(f"\nTotal JSON files: {total_files}, ~{int(0.8 * approximate_chunk_count)}-{approximate_chunk_count} chunks\n")
    else:
        approximate_chunk_count = sum(1 for _ in open(os.path.join(input_dir, jsonl_files[0]), 'r', encoding='utf-8'))

    counter_lock = threading.Lock()
    shared_counters = {"success": 0, "failure": 0}
    file_results = {}

    if use_progress_bar:
        pbar_total = tqdm(total=approximate_chunk_count, desc=" Processed", position=0, unit="chunk")
        pbar_success = tqdm(total=approximate_chunk_count, desc="Successful", position=1, unit="chunk")
    else:
        pbar_total = config.NoOpTqdm()
        pbar_success = config.NoOpTqdm()

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        for filename in all_files:
            processed_file = os.path.join(output_dir, f'processed_{os.path.splitext(filename)[0]}.jsonl')
            file_path = os.path.join(input_dir, filename)
            if os.path.exists(processed_file) and not force:
                num_chunks = count_chunks_in_file(file_path, CHUNK_SIZE)
                config.logger.info(f"Skipping {filename} as processed file exists: {processed_file} ({num_chunks} chunks counted as successful)")
                pbar_total.update(num_chunks)
                pbar_success.update(num_chunks)
                with counter_lock:
                    shared_counters["success"] += num_chunks
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    if filename.lower().endswith(".json"):
                        json_str = file.read()
                        lines = [json_str]
                    else:
                        lines = file.readlines()
            except Exception as e:
                if config.shutdown_event.is_set():
                    config.logger.info("Shutdown in progress; suppressing error details.")
                    return (filename, None, None, None, False)
                else:
                    config.logger.error(f"Failed to read file {filename}: {e}")
                    continue

            for linenum, line in enumerate(lines, start=1):
                if config.shutdown_event.is_set():
                    config.logger.info("Shutdown in progress: stopping submission of new tasks (file loop).")
                    break
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
                for chunknum, chunk in enumerate(chunks, start=1):
                    if config.shutdown_event.is_set():
                        config.logger.info("Shutdown in progress: stopping submission of new tasks (chunk loop).")
                        break
                    future = executor.submit(process_chunk, model, filename, rec_path,
                                             linenum, chunknum, chunk,
                                             pbar_total, pbar_success,
                                             shared_counters, counter_lock)
                    futures.append(future)

        processed_chunks = {}  # Maps filename -> set of (linenum, chunknum) tuples
        for future in concurrent.futures.as_completed(futures):
            try:
                fname, linenum, chunknum, qa_pair, success = future.result(timeout=75)
                if qa_pair is not None:
                    chunk_id = (linenum, chunknum)
                    if fname not in processed_chunks:
                        processed_chunks[fname] = set()
                    if chunk_id not in processed_chunks[fname]:
                        file_results.setdefault(fname, []).append(qa_pair)
                        processed_chunks[fname].add(chunk_id)
            except TimeoutError:
                config.logger.warning("Chunk processing task timed out after 75s")
            except Exception as e:
                if config.shutdown_event.is_set():
                    config.logger.info("Shutdown in progress; suppressing error details.")
                    return (filename, None, None, None, False)
                else:
                    config.logger.error(f"Error processing a chunk: {e}")

    if use_progress_bar:
        remaining = pbar_total.total - pbar_total.n
        if remaining > 0:
            pbar_total.update(remaining)
        pbar_total.close()
        pbar_success.close()

    # Write out MCQs in JSONL format.
    os.makedirs(output_dir, exist_ok=True)
    for fname, qa_pairs in file_results.items():
        base = os.path.splitext(fname)[0]
        out_file = os.path.join(output_dir, f'processed_{base}.jsonl')
        config.logger.info(f"Writing {len(qa_pairs)} MCQs to {out_file}")
        try:
            if force:
                with config.output_file_lock:
                    with open(out_file, 'a', encoding='utf-8') as out_f:
                        for mcq in qa_pairs:
                            out_f.write(json.dumps(mcq, ensure_ascii=False) + "\n")
                new_count = len(qa_pairs)
            else:
                existing_mcqs = []
                existing_ids = set()
                if os.path.exists(out_file):
                    try:
                        with open(out_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    mcq = json.loads(line.strip())
                                    mcq_id = (mcq.get('file'), mcq.get('line'), mcq.get('chunk'), mcq.get('model'))
                                    if mcq_id not in existing_ids:
                                        existing_mcqs.append(mcq)
                                        existing_ids.add(mcq_id)
                                except json.JSONDecodeError:
                                    pass
                    except Exception as e:
                        if config.shutdown_event.is_set():
                            config.logger.info("Shutdown in progress; suppressing error details.")
                            return (fname, None, None, None, False)
                        else:
                            config.logger.warning(f"Error reading existing file {out_file}: {e}")
                new_mcqs = []
                for pair in qa_pairs:
                    mcq_id = (pair.get('file'), pair.get('line'), pair.get('chunk'), pair.get('model'))
                    if mcq_id not in existing_ids:
                        new_mcqs.append(pair)
                        existing_ids.add(mcq_id)
                all_mcqs = existing_mcqs + new_mcqs
                with config.output_file_lock:
                    with open(out_file, 'w', encoding='utf-8') as out_f:
                        for mcq in all_mcqs:
                            out_f.write(json.dumps(mcq, ensure_ascii=False) + "\n")
                new_count = len(new_mcqs)
            config.logger.info(f"Wrote {len(qa_pairs)} MCQs to {out_file} ({new_count} new)")
        except Exception as e:
            if config.shutdown_event.is_set():
                config.logger.info("Shutdown in progress; suppressing error details.")
                return (fname, None, None, None, False)
            else:
                config.logger.error(f"Failed to write output file {out_file}: {e}")

    file_results.clear()
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
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if output files exist (append new MCQs).')

    args = parser.parse_args()

    use_progress_bar = config.configure_verbosity(args)
    input_directory = args.input
    output_json = args.output
    model_name = args.model
    model = Model(model_name)
    model.details()

    os.makedirs(output_json, exist_ok=True)

    try:
        process_directory(model, input_directory, output_json,
                          use_progress_bar=use_progress_bar,
                          parallel_workers=args.parallel,
                          force=args.force)
    except KeyboardInterrupt:
        config.initiate_shutdown("User Interrupt - initiating shutdown.")

