#!/usr/bin/env python
"""
mcq_util.py

Utility functions for the MCQ generation workflow.
This module contains helper functions for file reading, chunk splitting,
robust JSON parsing, processing text chunks, merging output, and updating counters.

Functions:
    human_readable_time      (seconds: float) -> str:
    load_file_lines          (filepath: str) -> list:
    split_text_into_chunks   (text: str, chunk_size: int = CHUNK_SIZE) -> list:
    count_chunks_in_file     (filepath: str, chunk_size: int = CHUNK_SIZE) -> int:
    update_shared_counters   (is_success: bool, shared_counters: dict, counter_lock: threading.Lock):
    attempt_parse_json       (s: str) -> dict:
    robust_parse_json_output (response_text: str, model) -> dict:
    process_chunk            (model, filename, file_path, linenum, chunknum, chunk,
                              pbar_total, pbar_success, shared_counters, counter_lock):
    merge_mcq_output         (out_file: str, new_qa_pairs: list) -> list:
    process_directory        (model, input_dir: str, output_dir: str = "output_files",
                              use_progress_bar: bool = True, parallel_workers: int = 4,
                              force: bool = False):
"""

import os
import json
import re
import threading
import time
import spacy
from tqdm import tqdm
from concurrent.futures import TimeoutError
from common import config

# Global constant
CHUNK_SIZE = config.chunkSize

# Initialize spaCy model once.
nlp = spacy.load("en_core_web_sm")


def human_readable_time(seconds: float) -> str:
    """Return a human-readable string for the given number of seconds."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.2f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.2f} hours"
    else:
        return f"{seconds/86400:.2f} days"


def load_file_lines(filepath: str) -> list:
    """
    Read a file and return its lines.
    If it is a JSON file, return a single-element list with the entire content.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        if filepath.lower().endswith(".json"):
            return [file.read()]
        else:
            return file.readlines()

# quicker estimate of chunk count up front - saves 10-20s of startup and associated
# user head-scratching before pbar is displayed

def estimate_chunk_count(input_dir: str, files: list[str], bytes_per_chunk: int = 1000) -> int:
    """
    Cheaper heuristic: assume ~1 chunk per `bytes_per_chunk` bytes of file size.
    """
    total = 0
    for f in files:
        try:
            size = os.path.getsize(os.path.join(input_dir, f))
            total += max(1, size // bytes_per_chunk)
        except OSError:
            continue
    return total


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """
    Split text into chunks of roughly chunk_size words by using spaCy for sentence segmentation.
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


def count_chunks_in_file(filepath: str, chunk_size: int = CHUNK_SIZE) -> int:
    """
    Count the number of chunks in a file.
    Uses load_file_lines and split_text_into_chunks.
    """
    total_chunks = 0
    try:
        lines = load_file_lines(filepath)
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


def update_shared_counters(is_success: bool, shared_counters: dict, counter_lock: threading.Lock):
    """
    Update counters for successes and failures, and if the success rate falls below 50% (after a threshold),
    initiate shutdown.
    """
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
                    f"Success rate <50% ({shared_counters['success']}/{total}). Run with -v to investigate."
                )
                config.initiate_shutdown("Failure rate >50%. Exiting.")


def attempt_parse_json(s: str) -> dict:
    """
    Try to parse a JSON string. If the result is a string, parse it again.
    Raise an exception if the final result is not a dictionary.
    """
    parsed = json.loads(s)
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    if not isinstance(parsed, dict):
        raise ValueError(f"Parsed JSON is not an object: {parsed}")
    return parsed


def robust_parse_json_output(response_text: str, model) -> dict:
    """
    Robustly parse the JSON output from a model.
    If the initial attempt fails, log the error and use a fix prompt.
    """
    cleaned = response_text.replace("```json", "").replace("```", "").strip()
    if cleaned.isdigit():
        return {"answer": "", "score": int(cleaned), "comment": ""}
    m = re.match(r"^(\d+)\)\s*(.*)", cleaned)
    if m:
        return {"answer": m.group(2).strip(), "score": int(m.group(1)), "comment": ""}
    try:
        return attempt_parse_json(cleaned)
    except Exception as e:
        if config.shutdown_event.is_set():
            raise ValueError("Shutdown in progress")
        try:
            with open("json_error_log.txt", "a", encoding="utf-8") as error_file:
                error_file.write(f"Initial JSON parsing error: {e}\nResponse: {response_text}\n\n")
        except Exception as file_e:
            config.logger.error(f"Failed to write JSON error log: {file_e}")
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
            return attempt_parse_json(fixed_output)
        except Exception as fix_error:
            try:
                with open("json_error_log.txt", "a", encoding="utf-8") as error_file:
                    error_file.write(f"Fix JSON parsing error: {fix_error}\nFixed output: {fixed_output}\n\n")
            except Exception as file_e:
                config.logger.error(f"Failed to write fix error log: {file_e}")
            raise ValueError(f"Failed to reformat JSON output. Original error: {e}; fix error: {fix_error}")


def process_chunk(model, filename, file_path, linenum, chunknum, chunk,
                  pbar_total, pbar_success, shared_counters, counter_lock):
    """
    Process a single chunk in three sequential steps:
      1. Summarize & expand the chunk.
      2. Generate a multiple-choice question.
      3. Verify and score the result.
    Returns a tuple: (filename, linenum, chunknum, QA pair dict (or None), success_flag)
    """
    if config.shutdown_event.is_set():
        return (filename, linenum, chunknum, None, False)

    config.logger.info(f"Processing chunk {chunknum} in file {filename}.")
    qa_pair = None
    chunk_success = False

    # Step 1: Summarize & Expand.
    try:
        step1_msg = config.user_message.format(chunk=chunk)
        step1_output = model.run(user_prompt=step1_msg, system_prompt=config.system_message)
        if config.shutdown_event.is_set():  # Check after model.run
            return (filename, linenum, chunknum, None, False)
        augmented_chunk = step1_output
        # If the response contains "augmented_chunk:", split it.
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
        config.logger.warning(f"Error summarizing chunk {chunknum} in file {filename}: {e}")
        pbar_total.update(1)
        update_shared_counters(False, shared_counters, counter_lock)
        return (filename, linenum, chunknum, None, False)

    # Step 2: Generate the MCQ.
    try:
        step2_msg = config.user_message_2.format(augmented_chunk=augmented_chunk)
        generated_question = model.run(user_prompt=step2_msg, system_prompt=config.system_message_2)
        if config.shutdown_event.is_set():  # Check after model.run
            return (filename, linenum, chunknum, None, False)
    except Exception as e:
        if config.shutdown_event.is_set():
            config.logger.info("Shutdown in progress; suppressing error details.")
            return (filename, linenum, chunknum, None, False)
        config.logger.warning(f"Error generating question for chunk {chunknum} in file {filename}: {e}")
        pbar_total.update(1)
        update_shared_counters(False, shared_counters, counter_lock)
        return (filename, linenum, chunknum, None, False)

    # Step 3: Verify and Score the MCQ.
    try:
        step3_msg = config.user_message_3.format(
            augmented_chunk=augmented_chunk,
            generated_question=generated_question
        )
        step3_output = model.run(user_prompt=step3_msg, system_prompt=config.system_message_3)
        if config.shutdown_event.is_set():  # Check after model.run
            return (filename, linenum, chunknum, None, False)
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


def merge_mcq_output(out_file: str, new_qa_pairs: list) -> list:
    """
    Merge new QA pairs with any existing MCQs in out_file using a unique MCQ identifier.
    Returns a merged list of MCQ dictionaries.
    """
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
                        continue
        except Exception as e:
            config.logger.warning(f"Error reading existing file {out_file}: {e}")
    for pair in new_qa_pairs:
        mcq_id = (pair.get('file'), pair.get('line'), pair.get('chunk'), pair.get('model'))
        if mcq_id not in existing_ids:
            existing_mcqs.append(pair)
            existing_ids.add(mcq_id)
    return existing_mcqs


def process_directory(model, input_dir: str, output_dir: str = "output_files",
                      use_progress_bar: bool = True, parallel_workers: int = 4, force: bool = False):
    """
    Process all JSON/JSONL files in input_dir by scheduling each text chunk
    as a separate task. If output files already exist and force is False,
    those files are skipped.
    Writes MCQs to output_dir in JSONL format.
    """
    config.logger.info(f"Run with {parallel_workers} threads.")

    # Gather list of files.
    json_files  = [f for f in os.listdir(input_dir) if f.lower().endswith(".json")]
    jsonl_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jsonl")]
    all_files = json_files + jsonl_files
    total_files = len(all_files)

    if total_files == 0:
        config.logger.warning(f"No JSON files found in {input_dir}.")
        return

    overall_start_time = time.time()

    # Compute approximate chunk count using size heuristic
    if json_files:
        approximate_chunk_count = estimate_chunk_count(input_dir, all_files)
        config.logger.info(f"{total_files} JSON files, ~{approximate_chunk_count} chunks\n")
    else:
        # fallback for .jsonl
        approximate_chunk_count = sum(
            1 for _ in open(os.path.join(input_dir, jsonl_files[0]), 'r', encoding='utf-8')
        )

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
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        for filename in all_files:
            processed_file = os.path.join(output_dir, f'processed_{os.path.splitext(filename)[0]}.jsonl')
            file_path = os.path.join(input_dir, filename)
            if os.path.exists(processed_file) and not force:
                if config.shutdown_event.is_set():
                    #config.logger.info("Shutdown in progress; suppressing error details.")
                    return
                num_chunks = count_chunks_in_file(file_path, CHUNK_SIZE)
                config.logger.info(f"Skipping {filename}: {num_chunks} existing chunks counted as successful")
                pbar_total.update(num_chunks)
                pbar_success.update(num_chunks)
                with counter_lock:
                    shared_counters["success"] += num_chunks
                continue

            try:
                lines = load_file_lines(file_path)
            except Exception as e:
                if config.shutdown_event.is_set():
                    config.logger.info("Shutdown in progress; suppressing error details.")
                    return
                else:
                    config.logger.error(f"Failed to read file {filename}: {e}")
                    continue

            for linenum, line in enumerate(lines, start=1):
                if config.shutdown_event.is_set():
                    #config.logger.info("Shutdown in progress: stopping new tasks (file loop).")
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
                        config.logger.info("Shutdown in progress: stopping new tasks (chunk loop).")
                        break
                    # Only submit new task if shutdown hasn't been triggered
                    futures.append(executor.submit(process_chunk, model, filename, rec_path,
                                                    linenum, chunknum, chunk,
                                                    pbar_total, pbar_success,
                                                    shared_counters, counter_lock))
                    # Check immediately after submitting if shutdown was triggered
                    if config.shutdown_event.is_set():
                        config.logger.info("Shutdown triggered: stopping further task submission.")
                        break
        processed_chunks = {}  # filename -> set of (linenum, chunknum)
        for future in futures:
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
                    return
                else:
                    config.logger.error(f"Error processing a chunk: {e}")

    if use_progress_bar:
        remaining = pbar_total.total - pbar_total.n
        if remaining > 0:
            pbar_total.update(remaining)
        pbar_total.close()
        pbar_success.close()

    # Write out MCQs for each file.
    os.makedirs(output_dir, exist_ok=True)
    for fname, qa_pairs in file_results.items():
        base = os.path.splitext(fname)[0]
        out_file = os.path.join(output_dir, f'processed_{base}.jsonl')
        try:
            if force:
                with config.output_file_lock:
                    with open(out_file, 'a', encoding='utf-8') as out_f:
                        for mcq in qa_pairs:
                            out_f.write(json.dumps(mcq, ensure_ascii=False) + "\n")
                new_count = len(qa_pairs)
            else:
                merged = merge_mcq_output(out_file, qa_pairs)
                with config.output_file_lock:
                    with open(out_file, 'w', encoding='utf-8') as out_f:
                        for mcq in merged:
                            out_f.write(json.dumps(mcq, ensure_ascii=False) + "\n")
                new_count = len(qa_pairs)
            config.logger.info(f"Wrote {len(qa_pairs)} MCQs to {out_file} ({new_count} new)")
        except Exception as e:
            if config.shutdown_event.is_set():
                config.logger.info("Shutdown in progress; suppressing error details.")
                return
            else:
                config.logger.error(f"Failed to write output file {out_file}: {e}")

    file_results.clear()
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    if config.shutdown_event.is_set():
        config.logger.warning("Process terminated")
    else:
        config.logger.info(
            f"Processed {total_files} files in {human_readable_time(total_time)}.\n"
            f"      Chunks processed:: {shared_counters}\n"
            f"      MCQs saved to {output_dir}."
        )

