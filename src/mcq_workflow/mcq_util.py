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
import re
from tqdm import tqdm
from concurrent.futures import TimeoutError
from common.loader import load_settings
from common import config  # Keep for backward compatibility

# Initialize settings
settings = load_settings()

# Global constant - use settings but fall back to config for backward compatibility
CHUNK_SIZE = settings.quality.chunkSize if hasattr(settings, 'quality') else config.chunkSize

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

import os
import json
# Use the already imported settings
# CHUNK_SIZE already defined above

def estimate_chunk_count(input_dir: str, files: list[str], *_ignore) -> int:
    """
    Heuristic: count total words in each file and assume CHUNK_SIZE words per chunk.
    Any extra words count as one more chunk.
    """
    config.logger.info("Estimating chunks to process.")
    total = 0
    for fn in files:
        path = os.path.join(input_dir, fn)
        try:
            with open(path, 'r', encoding='utf-8') as fp:
                if fn.lower().endswith('.json'):
                    rec = json.load(fp)
                    text = rec.get('text', '')
                else:
                    text = fp.read()
            words = len(text.split())
            total += max(1, words // CHUNK_SIZE)
        except Exception:
            # on error (e.g. unreadable file), just skip it but let the user know as a FYI
            config.logger.info(f"estimate_chunk_count: skipping {fn} - could not read/parse: {e}")
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
        config.logger.error(f"Failed to count chunks in file {filepath}: {e}")  # Keep using config.logger for compatibility
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
                  pbar_total, pbar_success, shared_counters, counter_lock, 
                  num_answers: int = 4):
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
        # Get prompts from settings if available, otherwise fall back to config
        user_message = settings.prompts.get("user_message_1") if hasattr(settings, "prompts") else config.user_message
        system_message = settings.prompts.get("system_message_1") if hasattr(settings, "prompts") else config.system_message

        step1_msg = user_message.format(chunk=chunk)
        step1_output = model.run(user_prompt=step1_msg, system_prompt=system_message)
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
        config.logger.info(f"Error summarizing chunk {chunknum} in file {filename}: {e}")
        pbar_total.update(1)
        update_shared_counters(False, shared_counters, counter_lock)
        return (filename, linenum, chunknum, None, False)

    # Step 2: Generate the MCQ.
    try:
        # Get prompts from settings if available, otherwise fall back to config
        user_message_2 = settings.prompts.get("user_message_mcq_2") if hasattr(settings, "prompts") else config.user_message_2
        system_message_2 = settings.prompts.get("system_message_mcq_2") if hasattr(settings, "prompts") else config.system_message_2

        # Format prompts with num_answers parameter
        system_msg_2 = system_message_2.format(num_answers=num_answers)
        step2_msg = user_message_2.format(augmented_chunk=augmented_chunk, num_answers=num_answers)
        generated_question = model.run(user_prompt=step2_msg, system_prompt=system_msg_2)
        if config.shutdown_event.is_set():  # Check after model.run
            return (filename, linenum, chunknum, None, False)
        # find the correct answer with the '(*)' marker
        m = re.search(r'^\s*(\d+)\)[^\n]*\(\*\)', generated_question, flags=re.M)
        correct_choice = m.group(1) if m else ""
    except Exception as e:
        if config.shutdown_event.is_set():
            config.logger.info("Shutdown in progress; suppressing error details.")
            return (filename, linenum, chunknum, None, False)
        config.logger.info(f"Error generating question for chunk {chunknum} in file {filename}: {e}")
        pbar_total.update(1)
        update_shared_counters(False, shared_counters, counter_lock)
        return (filename, linenum, chunknum, None, False)

    # Step 3: Verify and Score the MCQ.
    try:
        # Get prompts from settings if available, otherwise fall back to config
        user_message_3 = settings.prompts.get("user_message_mcq_3") if hasattr(settings, "prompts") else config.user_message_3
        system_message_3 = settings.prompts.get("system_message_mcq_3") if hasattr(settings, "prompts") else config.system_message_3

        # Extract choices from the generated_question
        # The question typically contains the full text with question followed by numbered choices
        # We need to extract just the choices part
        question_parts = generated_question.split("\n")
        generated_choices = ""
        collecting_choices = False
        
        for line in question_parts:
            # Start collecting when we see a line starting with a number (like "1." or "1)")
            if re.match(r'^\d+[\.\)]', line.strip()):
                collecting_choices = True
            
            if collecting_choices:
                generated_choices += line + "\n"
        
        # If we couldn't extract choices, use the whole question as a fallback
        if not generated_choices:
            generated_choices = generated_question
            
        step3_msg = user_message_3.format(
            augmented_chunk=augmented_chunk,
            generated_question=generated_question,
            generated_choices=generated_choices
        )
        step3_output = model.run(user_prompt=step3_msg, system_prompt=system_message_3)
        if config.shutdown_event.is_set():  # Check after model.run
            return (filename, linenum, chunknum, None, False)
            
        # Check for None or empty responses from the model
        if step3_output is None or (isinstance(step3_output, str) and not step3_output.strip()):
            config.logger.warning(f"Empty response from model for chunk {chunknum} in file {filename}. This may be due to a timeout or API error.")
            # Default to a low score instead of raising an error
            parsed_json = {"answer": "", "score": 0, "comment": "No response from model"}
        else:
            # Process normally when we have a response
            step3_clean = step3_output.replace("```json", "").replace("```", "").strip()
            parsed_json = robust_parse_json_output(step3_clean, model)
        #model_answer = str(parsed_json.get("answer", "")).strip()
        model_score = parsed_json.get("score", 0)
        pbar_total.set_postfix_str(f"Score: {model_score}")

        # Get minScore from settings if available, otherwise fall back to config
        min_score = settings.quality.minScore if hasattr(settings, "quality") else config.minScore

        if isinstance(model_score, int) and model_score > min_score:
            config.logger.info(f"MCQ generated for chunk {chunknum} in file {filename}, score {model_score} > {min_score}.")
            qa_pair = {
                "file": filename,
                "path": file_path,
                "line": linenum,
                "chunk": chunknum,
                "model": model.model_name,
                "question": generated_question,
                "answer": correct_choice,
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
            config.logger.info(f"Error reading existing file {out_file}: {e}")
    for pair in new_qa_pairs:
        mcq_id = (pair.get('file'), pair.get('line'), pair.get('chunk'), pair.get('model'))
        if mcq_id not in existing_ids:
            existing_mcqs.append(pair)
            existing_ids.add(mcq_id)
    return existing_mcqs


def process_directory(
    model,
    input_dir: str,
    output_dir: str = "output_files",
    use_progress_bar: bool = True,
    parallel_workers: int = 4,
    force: bool = False,
    num_answers: int = 4,
):
    """
    Process all JSON/JSONL files in input_dir by scheduling each text chunk
    as a separate task. If output files already exist and force is False,
    those chunks are counted as successes rather than reprocessed.
    Writes MCQs to output_dir in JSONL format.
    """
    config.logger.info(f"Run with {parallel_workers} threads.")

    # Gather list of files.
    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".json")]
    jsonl_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jsonl")]
    all_files = json_files + jsonl_files
    total_files = len(all_files)
    if total_files == 0:
        config.logger.warning(f"No JSON files found in {input_dir}.")
        return

    overall_start_time = time.time()

    # Compute approximate chunk count using heuristic
    if json_files:
        approximate_chunk_count = estimate_chunk_count(input_dir, all_files)
        config.logger.info(f"{total_files} JSON files, ~{approximate_chunk_count} chunks\n")
    else:
        approximate_chunk_count = sum(
            1 for _ in open(os.path.join(input_dir, jsonl_files[0]), 'r', encoding='utf-8')
        )

    counter_lock = threading.Lock()
    shared_counters = {"success": 0, "failure": 0}
    file_results: dict[str, list] = {}
    processed_chunks: dict[str, set[tuple[int,int]]] = {}

    if use_progress_bar:
        pbar_total = tqdm(total=approximate_chunk_count, desc=" Processed", position=0, unit="chunk")
        pbar_success = tqdm(total=approximate_chunk_count, desc="Successful", position=1, unit="chunk")
    else:
        pbar_total = config.NoOpTqdm()
        pbar_success = config.NoOpTqdm()

    # ------------------------------------------------------------------------
    # Parallel execution with clean interrupt handling
    # ------------------------------------------------------------------------
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    executor = ThreadPoolExecutor(max_workers=parallel_workers)
    try:
        futures = []

        # 1) Submit tasks
        for filename in all_files:
            # If already processed and not forcing, skip
            processed_file = os.path.join(output_dir, f"processed_{os.path.splitext(filename)[0]}.jsonl")
            file_path = os.path.join(input_dir, filename)
            if os.path.exists(processed_file) and not force:
                if config.shutdown_event.is_set():
                    return
                num_chunks = count_chunks_in_file(file_path, CHUNK_SIZE)
                config.logger.info(f"Skipping {filename}: {num_chunks} existing chunks")
                pbar_total.update(num_chunks)
                pbar_success.update(num_chunks)
                with counter_lock:
                    shared_counters["success"] += num_chunks
                continue

            # Read file lines
            try:
                lines = load_file_lines(file_path)
            except Exception as e:
                if config.shutdown_event.is_set():
                    return
                config.logger.error(f"Failed to read file {filename}: {e}")
                continue

            # Break into chunks and submit
            for linenum, line in enumerate(lines, start=1):
                if config.shutdown_event.is_set():
                    break
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    config.logger.info(f"JSON decode error in {filename} line {linenum}: {e}")
                    continue
                text = record.get("text", "")
                rec_path = record.get("path", file_path)
                if not text:
                    continue
                chunks = split_text_into_chunks(text, CHUNK_SIZE)

                for chunknum, chunk in enumerate(chunks, start=1):
                    if config.shutdown_event.is_set():
                        break
                    futures.append(
                        executor.submit(
                            process_chunk,
                            model,
                            filename,
                            rec_path,
                            linenum,
                            chunknum,
                            chunk,
                            pbar_total,
                            pbar_success,
                            shared_counters,
                            counter_lock,
                            num_answers,
                        )
                    )
                    if config.shutdown_event.is_set():
                        break

        # 2) Collect results
        for fut in futures:
            try:
                fname, linenum, chunknum, qa_pair, success = fut.result(timeout=settings.timeout)
                if qa_pair:
                    chunk_id = (linenum, chunknum)
                    if fname not in processed_chunks:
                        processed_chunks[fname] = set()
                    if chunk_id not in processed_chunks[fname]:
                        file_results.setdefault(fname, []).append(qa_pair)
                        processed_chunks[fname].add(chunk_id)
            except TimeoutError:
                config.logger.info("Chunk processing task timed out")
            except Exception as e:
                if config.shutdown_event.is_set():
                    return
                config.logger.error(f"Error processing a chunk: {e}")

    except KeyboardInterrupt:
        config.logger.warning("Interrupt received – cancelling pending tasks…")
        executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)
    else:
        executor.shutdown()

    # Finalize progress bars
    if use_progress_bar:
        remaining = pbar_total.total - pbar_total.n
        if remaining > 0:
            pbar_total.update(remaining)
        pbar_total.close()
        pbar_success.close()

    # Write out MCQs for each file
    os.makedirs(output_dir, exist_ok=True)
    for fname, qa_pairs in file_results.items():
        base = os.path.splitext(fname)[0]
        out_file = os.path.join(output_dir, f"processed_{base}.jsonl")
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
                return
            config.logger.error(f"Failed to write output file {out_file}: {e}")

    # Cleanup and summary
    file_results.clear()
    end_time = time.time()
    elapsed = end_time - overall_start_time
    if config.shutdown_event.is_set():
        config.logger.warning("Process terminated by interrupt")
    else:
        config.logger.info(
            f"Processed {total_files} files in {human_readable_time(elapsed)}. "
            f"Chunks: {shared_counters}"
        )

