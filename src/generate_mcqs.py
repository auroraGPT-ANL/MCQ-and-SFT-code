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
import argparse
from tqdm import tqdm  # CeC: tqdm for progress bar

from model_access import Model


##############################################################################
# Global constants
##############################################################################
CHUNK_SIZE = config.chunkSize
chunks_successful = 0
chunks_failed     = 0

nlp = spacy.load("en_core_web_sm") 


######
# Functions
######

def human_readable_time(seconds: float) -> str:
    """
    Convert time in seconds into a more human-friendly format
    (seconds, minutes, hours, days).
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

def generate_mcqs(model, path, filename, linenum, chunks: list, pbar) -> list:
    """
    For each chunk:
      1) Summarize and expand the chunk => augmented_chunk
      2) Generate a MULTIPLE-CHOICE question with 5 possible answers,
         marking the correct answer for later grading.
      3) Verify by prompting GPT with (question + augmented_chunk) to get
         an answer & a score (1-10). Keep only those with score > 7.

    Returns a list of dicts with the QA pair if they pass the score threshold.
    """

    global chunks_successful
    global chunks_failed
    qa_pairs = []

    for chunknum, chunk in enumerate(chunks, start=1):
        # Step 1: Summarize & expand the chunk
        try:
            formatted_user_message = config.user_message.format(chunk=chunk)

            step1_output = model.run(user_prompt=formatted_user_message,
                                     system_prompt= config.system_message)
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
            # Check for fatal errors
            if "401" in str(e) or "Unauthorized" in str(e):
                sys.exit(f"Model API Authentication failed. ({str(e)}) Exiting.")
            pbar.update(1)
            chunks_failed +=1
            continue

        # Step 2: Generate a MULTIPLE-CHOICE question with 5 answers
        try:
            formatted_user_message_2 = config.user_message_2.format(augmented_chunk=augmented_chunk)
            generated_question = model.run( user_prompt=formatted_user_message_2,
                                           system_prompt=config.system_message_2
            )
        except Exception as e:
            config.logger.warning(f"Error generating question: {e}")
            # Check for fatal errors
            if "401" in str(e) or "Unauthorized" in str(e):
                sys.exit("Model API Authentication failed. Exiting.")
            pbar.update(1)
            chunks_failed +=1
            continue

        # Step 3: Verify the question by prompting GPT with the augmented_chunk
        try:
            # Format the user prompt with both augmented_chunk and generated_question.
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

            # Update the progress bar with the current score.
            pbar.set_postfix_str(f"Score: {model_score}")

            if isinstance(model_score, int) and model_score > config.minScore:
                config.logger.info(f"mcq generated, score {model_score} > {config.minScore}.")
                chunks_successful += 1


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
                chunks_failed +=1
                config.logger.info(f"Chunk fail: could not fix JSON.")



        except json.JSONDecodeError:
            config.logger.info("Chunk JSON parsing failed. Trying to fix.")
            fix_prompt = f"""
            Convert the following text strictly into valid JSON with three key/value
            pairs: question, answer, score.  Nothing else, no additional text.

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
                    if "Expecting value: line 1 column 1" in str(e):
                        config.logger.info(f"Chunk fail: Output empty or not valid JSON: {e}")
                    else:
                        config.logger.info(f"Chunk fail: JSON decoding error: {e}")
                    continue

                model_answer = parsed_json.get("answer", "").strip()
                model_score = parsed_json.get("score", 0)
                pbar.set_postfix_str(f"Score: {model_score}")

                if isinstance(model_score, int) and model_score > config.minScore:
                    qa_pairs.append({
                        "question": generated_question,
                        "answer": model_answer,
                        "text": augmented_chunk
                    })
                    chunks_successful +=1
                else:
                    config.logger.info("Chunk fail: Could not fix JSON")
                    chunks_failed +=1

            except Exception as e:
                config.logger.info(f"Chunk fail: Could not fix JSON automatically: {e}")
                pbar.update(1)
                chunks_failed +=1
                continue

        except Exception as e:
            config.logger.info(f"Chunk fail: Error in verifying question/answer: {e}")
            pbar.update(1)
            chunks_failed +=1
            continue

        # Update the progress bar after processing this chunk
        pbar.update(1)

    return qa_pairs

def process_directory(model, input_dir: str, output_dir: str = "output_files", use_progress_bar: bool = True):

    """
    Main function to:
    1) Iterate over all JSON/JSONL files in a directory.
    2) Extract text.
    3) Split into chunks.
    4) Generate Q/A pairs (including summarization & scoring).
    5) Save to JSON only those that pass the score threshold.
    """
    json_files  = [f for f in os.listdir(input_dir) if f.lower().endswith(".json")]
    jsonl_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jsonl")]
    all_files = json_files + jsonl_files
    total_files = len(all_files)

    if total_files == 0:
        config.logger.warning("No suitable files found in directory.")
        return

    overall_start_time = time.time()
    cumulative_time   = 0.0
    processed_count   = 0

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
        # Fallback if only JSONL files exist (estimate by summing lines)
        approximate_chunk_count = sum(line_counts)

    # Create a global tqdm progress bar in default modefor chunk processing (suppress if in -v or -q mode))
    if use_progress_bar:
        pbar = tqdm(total=approximate_chunk_count, desc="Chunks processed", unit="chunk")
    else:
        pbar = config.NoOpTqdm()



    # Iterate over files
    for i, filename in enumerate(all_files, start=1):
        all_prompt_answer_pairs = []
        num_chunks = 0
        file_path = os.path.join(input_dir, filename)
        file_start_time = time.time()

        config.logger.info(f"Processing file {i}/{total_files}: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as file:
            if filename.lower().endswith(".json"):
                json_str = file.read()
                lines = [json_str]
            else:
                lines = file.readlines()

            for j, line in enumerate(lines, start=1):
                # JSON file will get read as one line most of the time so this msg is
                # misleading and not helpful
                #config.logger.info(f"Processing line {j} of {len(lines)} in file {i}")
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    config.logger.info(f"JSON decode error in file {filename} line {j}: {e}")
                    continue

                text = record['text']
                path = record['path']
                chunks = split_text_into_chunks(text, CHUNK_SIZE)
                num_chunks += len(chunks)

                prompt_answer_pairs = generate_mcqs(model, path, filename, j, chunks, pbar)
                all_prompt_answer_pairs.extend(prompt_answer_pairs)

        out_file = f'{output_dir}/file_{i}.json'
        config.logger.info(f"Writing output for file {i} with {num_chunks} chunks to {out_file}")
        with open(out_file, 'w', encoding='utf-8') as out_f:
            json.dump(all_prompt_answer_pairs, out_f, ensure_ascii=False, indent=2)

        file_end_time = time.time()
        file_time_taken = file_end_time - file_start_time
        processed_count += 1
        cumulative_time += file_time_taken
        avg_time_per_file_so_far = cumulative_time / processed_count
        remaining_files = total_files - processed_count
        estimated_time_remaining = remaining_files * avg_time_per_file_so_far

        config.logger.info(
            f"Time for this file: {human_readable_time(file_time_taken)}. | "
            f"Processed: {processed_count}/{total_files}. | "
            f"Estimated remaining: {human_readable_time(estimated_time_remaining)}"
        )

    total_time = time.time() - overall_start_time
    config.logger.info(
        f"Processed file in "
        f"{human_readable_time(total_time)}.\n      "
        f"{chunks_successful} MCQ's saved, {chunks_failed} failed.  | "
        f"Prompt/answer pairs (score > 7) saved to {output_dir}."
    )

    if processed_count > 0:
        final_avg_time_per_file = total_time / processed_count
        config.logger.info(f"Average time to process each file: {human_readable_time(final_avg_time_per_file)}")

    # Final logging (after processing all files)
    total_time = time.time() - overall_start_time
    config.logger.info(
        f"Done. Processed {processed_count}/{total_files} files in "
        f"{human_readable_time(total_time)}.\n      "
        f"{chunks_successful} chunks succeeded, {chunks_failed} failed.\n       "
        f"Prompt/answer pairs (score > 7) saved to {output_dir}."
    )
    if processed_count > 0:
        final_avg_time_per_file = total_time / processed_count
        config.logger.info(f"Average time to process each file: {human_readable_time(final_avg_time_per_file)}")

    # Force the progress bar to complete by updating it to the total if necessary
    remaining = pbar.total - pbar.n
    if remaining > 0:
        pbar.update(remaining)
    # Close the progress bar
    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Program to generate MCQs from JSONL or JSON files')
    parser.add_argument('-i', '--input',  help='Directory containing input JSON/JSONL files',
                        default=config.json_dir)  
    parser.add_argument('-o', '--output', help='Output directory for MCQs',
                        default=config.mcq_dir)
    parser.add_argument('-m','--model', help='Model to use to generate MCQs',
                        default=config.defaultModel)
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

    input_directory = args.input
    output_json     = args.output

    model_name = args.model
    model = Model(model_name)
    model.details()

    os.makedirs(output_json, exist_ok=True)

    try:
        process_directory(model, input_directory, output_json, use_progress_bar=use_progress_bar)

    except KeyboardInterrupt:
        print("EXIT: Execution interrupted by user")
        sys.exit(0)

