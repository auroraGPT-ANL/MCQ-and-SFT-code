#!/usr/bin/env python

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
from tqdm import tqdm  # CeC: tqdm for progress bar
import concurrent.futures

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

def approximate_total_chunks(input_dir, bytes_per_chunk=CHUNK_SIZE):
    """
    Returns an approximate total chunk count by summing file sizes
    of .json or .jsonl files and dividing by `bytes_per_chunk`.
    """
    total_bytes = 0
    for f in os.listdir(input_dir):
        if f.lower().endswith((".json", ".jsonl")):
            path = os.path.join(input_dir, f)
            size = os.stat(path).st_size  # file size in bytes
            total_bytes += size
    return total_bytes // bytes_per_chunk

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

def generate_mcqs(model, path, filename, linenum, chunks: list, pbar) -> tuple:
    """
    For each chunk:
      1) Summarize and expand the chunk.
      2) Generate a multiple-choice question with 5 possible answers.
      3) Verify by prompting GPT (question + augmented chunk) and check the score.

    Returns a tuple: (qa_pairs, local_success, local_failed)
    """
    qa_pairs = []
    local_success = 0
    local_failed = 0

    for chunknum, chunk in enumerate(chunks, start=1):
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
            config.logger.warning(f"Error summarizing and expanding chunk: {e}")
            if "401" in str(e) or "Unauthorized" in str(e):
                sys.exit(f"Model API Authentication failed. ({str(e)}) Exiting.")
            pbar.update(1)
            local_failed += 1
            continue

        # Step 2: Generate the multiple-choice question
        try:
            formatted_user_message_2 = config.user_message_2.format(augmented_chunk=augmented_chunk)
            generated_question = model.run(user_prompt=formatted_user_message_2,
                                           system_prompt=config.system_message_2)
        except Exception as e:
            config.logger.warning(f"Error generating question: {e}")
            if "401" in str(e) or "Unauthorized" in str(e):
                sys.exit("Model API Authentication failed. Exiting.")
            pbar.update(1)
            local_failed += 1
            continue

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
                raise ValueError("Chunk Fail: model.run() returned None for step3_output.")

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
            pbar.set_postfix_str(f"Score: {model_score}")

            if isinstance(model_score, int) and model_score > config.minScore:
                config.logger.info(f"MCQ generated, score {model_score} > {config.minScore}.")
                local_success += 1
                qa_pairs.append({
                    "file": filename,
                    "path": path,
                    "line": linenum,
                    "chunk": chunknum,
                    "model": model.model_name,
                    "question": generated_question,
                    "answer": model_answer,
                    "text": augmented_chunk
                })
            else:
                local_failed += 1

        except json.JSONDecodeError:
            config.logger.info("Chunk JSON parsing failed. Trying to fix.")
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
                try:
                    parsed_json = json.loads(fixed_json_output)
                    if isinstance(parsed_json, str):
                        parsed_json = json.loads(parsed_json)
                except json.JSONDecodeError as e:
                    config.logger.info(f"Chunk fail: Output empty or not valid JSON: {e}")
                    pbar.update(1)
                    local_failed += 1
                    continue

                model_answer = parsed_json.get("answer", "").strip()
                model_score = parsed_json.get("score", 0)
                pbar.set_postfix_str(f"Score: {model_score}")

                if isinstance(model_score, int) and model_score > config.minScore:
                    qa_pairs.append({
                        "file": filename,
                        "path": path,
                        "line": linenum,
                        "chunk": chunknum,
                        "model": model.model_name,
                        "question": generated_question,
                        "answer": model_answer,
                        "text": augmented_chunk
                    })
                    local_success += 1
                else:
                    config.logger.info("Chunk fail: Could not fix JSON")
                    local_failed += 1

            except Exception as e:
                config.logger.info(f"Chunk fail: Could not fix JSON automatically: {e}")
                pbar.update(1)
                local_failed += 1
                continue

        except Exception as e:
            config.logger.info(f"Chunk fail: Error in verifying question/answer: {e}")
            pbar.update(1)
            local_failed += 1
            continue

        pbar.update(1)
    return qa_pairs, local_success, local_failed

def process_file(model, filename, file_index, input_dir, output_dir, pbar, total_files):
    """
    Process a single file: read, split into chunks, generate MCQs, and write output.
    Returns a tuple: (local_success_total, local_failed_total, file_time_taken, num_chunks)
    """
    all_prompt_answer_pairs = []
    num_chunks = 0
    file_path = os.path.join(input_dir, filename)
    file_start_time = time.time()

    config.logger.info(f"Processing file {file_index}/{total_files}: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            if filename.lower().endswith(".json"):
                json_str = file.read()
                lines = [json_str]
            else:
                lines = file.readlines()
    except Exception as e:
        config.logger.error(f"Failed to read file {filename}: {e}")
        return (0, 0, 0, 0)

    local_success_total = 0
    local_failed_total = 0

    for j, line in enumerate(lines, start=1):
        try:
            record = json.loads(line.strip())
        except json.JSONDecodeError as e:
            config.logger.info(f"JSON decode error in file {filename} line {j}: {e}")
            continue

        text = record['text']
        path = record['path']
        chunks = split_text_into_chunks(text, CHUNK_SIZE)
        num_chunks += len(chunks)

        qa_pairs, success, failed = generate_mcqs(model, path, filename, j, chunks, pbar)
        local_success_total += success
        local_failed_total += failed
        all_prompt_answer_pairs.extend(qa_pairs)

    out_file = os.path.join(output_dir, f'file_{file_index}.json')
    config.logger.info(f"Writing output for file {file_index} with {num_chunks} chunks to {out_file}")
    try:
        with open(out_file, 'w', encoding='utf-8') as out_f:
            json.dump(all_prompt_answer_pairs, out_f, ensure_ascii=False, indent=2)
    except Exception as e:
        config.logger.error(f"Failed to write output file {out_file}: {e}")

    file_end_time = time.time()
    file_time_taken = file_end_time - file_start_time
    config.logger.info(
        f"Time for file {file_index}: {human_readable_time(file_time_taken)} | "
        f"Chunks processed: {num_chunks} | Success: {local_success_total} | Failed: {local_failed_total}"
    )
    return (local_success_total, local_failed_total, file_time_taken, num_chunks)

def process_directory(model, input_dir: str, output_dir: str = "output_files", use_progress_bar: bool = True, parallel_workers: int = 4):
    """
    Main function to process all JSON/JSONL files in parallel.
    """
    json_files  = [f for f in os.listdir(input_dir) if f.lower().endswith(".json")]
    jsonl_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jsonl")]
    all_files = json_files + jsonl_files
    total_files = len(all_files)

    if total_files == 0:
        config.logger.warning("No suitable files found in directory.")
        return

    overall_start_time = time.time()
    cumulative_time = 0.0

    if len(jsonl_files) > 0:
        line_counts = []
        for i, filename in enumerate(jsonl_files, start=1):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                line_count = sum(1 for _ in file)
            line_counts.append(line_count)
        config.logger.info(f'{len(jsonl_files)} JSONL files, with {sum(line_counts)} lines in total: {line_counts}')

    if len(json_files) > 0:
        approximate_chunk_count = approximate_total_chunks(input_dir, bytes_per_chunk=CHUNK_SIZE)
        config.logger.info(f"\nTotal JSON files: {total_files}, "
                           f"~{int(0.8 * approximate_chunk_count)}-{approximate_chunk_count} chunks\n")
    else:
        approximate_chunk_count = sum(line_counts)

    if use_progress_bar:
        pbar = tqdm(total=approximate_chunk_count, desc="Chunks processed", unit="chunk")
    else:
        pbar = config.NoOpTqdm()

    os.makedirs(output_dir, exist_ok=True)

    total_success = 0
    total_failed = 0
    total_chunks = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        future_to_file = {
            executor.submit(process_file, model, filename, i, input_dir, output_dir, pbar, total_files): filename
            for i, filename in enumerate(all_files, start=1)
        }
        for future in concurrent.futures.as_completed(future_to_file):
            try:
                success, failed, file_time, num_chunks = future.result()
                total_success += success
                total_failed += failed
                total_chunks += num_chunks
                cumulative_time += file_time
            except Exception as e:
                config.logger.error(f"Error processing a file: {e}")

    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    config.logger.info(
        f"Processed {total_files} files in {human_readable_time(total_time)}.\n"
        f"{total_success} MCQ's succeeded, {total_failed} failed. | "
        f"Prompt/answer pairs saved to {output_dir}."
    )
    if total_files > 0:
        final_avg_time_per_file = total_time / total_files
        config.logger.info(f"Average time per file: {human_readable_time(final_avg_time_per_file)}")

    remaining = pbar.total - pbar.n
    if remaining > 0:
        pbar.update(remaining)
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Program to generate MCQs from JSONL or JSON files')
    parser.add_argument('-i', '--input',  help='Directory containing input JSON/JSONL files',
                        default=config.json_dir)
    parser.add_argument('-o', '--output', help='Output directory for MCQs',
                        default=config.mcq_dir)
    parser.add_argument('-m', '--model', help='Model to use to generate MCQs',
                        default=config.defaultModel)
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='No progress bar or messages')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('-p', '--parallel', type=int, default=4,
                        help='Number of parallel threads (default: 4)')

    args = parser.parse_args()

    use_progress_bar = config.configure_verbosity(args)

    input_directory = args.input
    output_json = args.output

    model_name = args.model
    model = Model(model_name)
    model.details()

    try:
        process_directory(model, input_directory, output_json, use_progress_bar=use_progress_bar, parallel_workers=args.parallel)
    except KeyboardInterrupt:
        print("EXIT: Execution interrupted by user")
        sys.exit(0)

