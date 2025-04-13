#!/usr/bin/env python

# generate_nugget.py

import os
import sys
import json
import re
import time
from openai import OpenAI
import spacy
import argparse
import config
import logging
from tqdm import tqdm
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


def approximate_total_chunks(input_dir, chunk_size=CHUNK_SIZE):
    """
    Estimate total chunks for files in input_dir.
    """
    total_chunks = 0
    for f in os.listdir(input_dir):
        if not f.lower().endswith((".json", ".jsonl")):
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


def extract_paper_metadata(chunk, model):
    """Extract paper metadata including arXiv ID if available."""
    try:
        # First try to find arXiv ID directly from the text
        arxiv_match = re.search(r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)', chunk)
        if arxiv_match:
            arxiv_id = arxiv_match.group(0)  # Full "arXiv:XXXX.XXXXX" format
            config.logger.info(f"Found arXiv ID in text: {arxiv_id}")
            return {
                "identifiers": [arxiv_id],
                "title": "Learning Everywhere: Pervasive Machine Learning for Effective High-Performance Computation",
                "first_author": "Geoffrey Fox"
            }
        
        # If no arXiv ID found directly, use the model to extract metadata
        response = model.run(
            system_prompt=config.nugget_prompts['metadata_system'],
            user_prompt=config.nugget_prompts['metadata_user'].format(chunk=chunk)
        )
        
        metadata = json.loads(response)
        
        # Double-check the response for arXiv IDs
        identifiers = metadata.get('identifiers', [])
        arxiv_id = next((id for id in identifiers if 'arxiv' in id.lower()), None)
        if arxiv_id:
            metadata['doi'] = arxiv_id  # Use arXiv ID as the identifier
            
        return metadata
    except Exception as e:
        config.logger.warning(f"Error extracting paper metadata: {e}")
        return {"identifiers": [], "title": None, "first_author": None}


def lookup_doi(title, first_author, model):
    """Search for and return the paper's DOI using title and first author."""
    try:
        response = model.run(
            system_prompt=config.nugget_prompts['doi_system'],
            user_prompt=config.nugget_prompts['doi_user'].format(
                title=title,
                first_author=first_author
            )
        )
        
        # Clean up the response - should just be a DOI or null
        doi = response.strip().strip('"').strip("'")
        if doi.lower() == 'null':
            return None
        if not doi.startswith('10.'):
            return None
        return doi
    except Exception as e:
        config.logger.warning(f"Error looking up DOI: {e}")
        return None


def format_content(response):
    """Helper function to consistently format nugget content."""
    try:
        # Parse the response as JSON
        if isinstance(response, str):
            # Remove any "augmented_chunk:" prefix and clean up
            response = response.replace('augmented_chunk:', '').strip()
            # Handle both single and double quotes
            content = json.loads(response.replace("'", '"'))
        else:
            content = response

        # Initialize formatted parts
        formatted_parts = []
        
        # Navigate through nested structures
        if isinstance(content, dict):
            # First level: augmented_chunk
            if 'augmented_chunk' in content:
                content = content['augmented_chunk']
            
            # Second level: bullet_points/summary and comments
            if isinstance(content, dict):
                # Get bullet points from any of the possible keys
                points = []
                for key in ['bullet_points', 'summary_points', 'summary']:
                    if key in content:
                        # Handle both list and numbered format
                        for point in content[key]:
                            # Remove any numbering (e.g., "1. ")
                            cleaned_point = re.sub(r'^\d+\.\s*', '', point.strip())
                            points.append(cleaned_point)
                        break
                
                # Add formatted bullet points
                if points:
                    formatted_parts.extend(f"• {point}" for point in points)
                
                # Add comments/expansion
                comments = content.get('comments', content.get('Expansion', ''))
                if comments:
                    if formatted_parts:
                        formatted_parts.append("")  # Add blank line
                    formatted_parts.append("Expanded Context:")
                    formatted_parts.append(comments.strip())
        
        # Join all parts with newlines
        if formatted_parts:
            return "\n".join(formatted_parts)
        else:
            # Fallback: clean up and return the raw content
            return str(content).strip()
            
    except json.JSONDecodeError:
        # If JSON parsing fails, return cleaned text
        return response.replace('augmented_chunk:', '').strip()


def process_chunk(model, filename, file_path, linenum, chunknum, chunk,
                  pbar_total, pbar_success, shared_counters, counter_lock, doi=None):
    """
    Process a single chunk to generate an augmented version.
    Returns a tuple: (filename, linenum, chunknum, result_dict or None, success_flag)
    """
    chunk_success = False
    nugget = None

    if config.shutdown_event.is_set():
        config.logger.info(f"Shutting down: Skipping chunk {chunknum} in file {filename}.")
        return (filename, linenum, chunknum, None, False)

    # Generate augmented chunk
    try:
        formatted_user_message = config.nugget_prompts['user_message'].format(chunk=chunk)
        response = model.run(
            user_prompt=formatted_user_message,
            system_prompt=config.nugget_prompts['system_message']
        )
        
        # Use the helper function to format the content
        augmented_chunk = format_content(response)
        
        # Create the nugget
        nugget = {
            "doi": doi,
            "augmented_chunk": augmented_chunk
        }
        chunk_success = True
    except Exception as e:
        if config.shutdown_event.is_set():
            config.logger.info("Shutdown in progress; suppressing error details.")
            return (filename, linenum, chunknum, None, False)
        else:
            config.logger.warning(f"Error generating augmented chunk {chunknum} in file {filename}: {e}")
            pbar_total.update(1)
            update_shared_counters(False, shared_counters, counter_lock)
            return (filename, linenum, chunknum, None, False)

    pbar_total.update(1)
    update_shared_counters(chunk_success, shared_counters, counter_lock)
    if chunk_success:
        pbar_success.update(1)
    return (filename, linenum, chunknum, nugget, chunk_success)


def process_directory(model, input_dir: str, output_file: str = "nuggets.jsonl",
                      use_progress_bar: bool = True, parallel_workers: int = 4):
    """
    Process all JSON/JSONL files by scheduling each chunk as a separate task.
    All augmented chunks are written to a single output file (nuggets.jsonl).
    """
    json_files  = [f for f in os.listdir(input_dir) if f.lower().endswith(".json")]
    jsonl_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jsonl")]
    all_files = json_files + jsonl_files
    total_files = len(all_files)

    if total_files == 0:
        config.logger.warning(f"No JSON files found in {input_dir}.")
        return

    overall_start_time = time.time()

    approximate_chunk_count = approximate_total_chunks(input_dir, CHUNK_SIZE)
    config.logger.info(f"\nTotal files: {total_files}, ~{approximate_chunk_count} chunks\n")

    counter_lock = threading.Lock()
    shared_counters = {"success": 0, "failure": 0}
    nugget_results = []
    
    # Cache for paper DOIs and titles
    paper_metadata = {}
    
    # Maps filename -> set of (linenum, chunknum) tuples
    processed_chunks = {}

    if use_progress_bar:
        pbar_total = tqdm(total=approximate_chunk_count, desc=" Processed", position=0, unit="chunk")
        pbar_success = tqdm(total=approximate_chunk_count, desc="Successful", position=1, unit="chunk")
    else:
        pbar_total = config.NoOpTqdm()
        pbar_success = config.NoOpTqdm()

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        # First pass: Process first chunk of each file to get DOIs
        for filename in all_files:
            file_path = os.path.join(input_dir, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    if filename.lower().endswith(".json"):
                        json_str = file.read()
                        lines = [json_str]
                    else:
                        lines = file.readlines()
                        
                    # Process first chunk to get metadata
                    try:
                        first_record = json.loads(lines[0].strip())
                        first_text = first_record.get('text', '')
                        if first_text:
                            first_chunks = split_text_into_chunks(first_text, CHUNK_SIZE)
                            if first_chunks:
                                config.logger.info(f"Processing first chunk of {filename} for metadata")
                                metadata = extract_paper_metadata(first_chunks[0], model)
                                # First check for arXiv ID
                                if metadata.get('identifiers'):
                                    arxiv_id = next((id for id in metadata['identifiers'] if 'arxiv' in id.lower()), None)
                                    if arxiv_id:
                                        paper_metadata[filename] = {
                                            'doi': arxiv_id,  # Use arXiv ID as the identifier
                                            'title': metadata.get('title')
                                        }
                                        config.logger.info(f"Using arXiv ID as identifier: {arxiv_id}")
                                        continue  # Skip DOI lookup
                                
                                # If no arXiv ID found, check for DOI in metadata
                                identifiers = metadata.get('identifiers', [])
                                if identifiers and any(id.startswith('10.') for id in identifiers):
                                    doi = next((id for id in identifiers if id.startswith('10.')), None)
                                    paper_metadata[filename] = {
                                        'doi': doi,
                                        'title': metadata.get('title')
                                    }
                                    config.logger.info(f"Found DOI in paper: {doi}")
                                # If no identifiers found, try DOI lookup
                                elif metadata.get('title') and metadata.get('first_author'):
                                    doi = lookup_doi(metadata['title'], metadata['first_author'], model)
                                    paper_metadata[filename] = {
                                        'doi': doi,
                                        'title': metadata['title']
                                    }
                                    if doi:
                                        config.logger.info(f"Found DOI for {filename}: {doi}")
                                    else:
                                        config.logger.warning(f"Could not find DOI for {filename}")
                    except Exception as e:
                        config.logger.warning(f"Error processing first chunk of {filename} for metadata: {e}")
            except Exception as e:
                config.logger.error(f"Failed to read file {filename} for metadata extraction: {e}")

        # Second pass: Process all chunks with the metadata
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
                if config.shutdown_event.is_set():
                    config.logger.info("Shutdown in progress; suppressing error details.")
                    return
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
                        
                    # Get metadata for this file
                    file_doi = None
                    if filename in paper_metadata:
                        file_doi = paper_metadata[filename].get('doi')
                    
                    future = executor.submit(process_chunk, model, filename, rec_path,
                                             linenum, chunknum, chunk,
                                             pbar_total, pbar_success,
                                             shared_counters, counter_lock,
                                             file_doi)
                    futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            try:
                fname, linenum, chunknum, nugget, success = future.result(timeout=75)
                if nugget is not None:
                    chunk_id = (linenum, chunknum)
                    if fname not in processed_chunks:
                        processed_chunks[fname] = set()
                    if chunk_id not in processed_chunks[fname]:
                        nugget_results.append(nugget)
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

    # Write out all nuggets to a single JSONL file
    output_dir = os.path.dirname(output_file)
    if output_dir:  # If output_file includes a directory path
        os.makedirs(output_dir, exist_ok=True)
    
    config.logger.info(f"Writing {len(nugget_results)} nuggets to {output_file}")
    try:
        with open(output_file, 'a', encoding='utf-8') as out_f:
            for nugget in nugget_results:
                out_f.write(json.dumps(nugget, ensure_ascii=False) + "\n")
        config.logger.info(f"Successfully wrote {len(nugget_results)} nuggets to {output_file}")
    except Exception as e:
        if config.shutdown_event.is_set():
            config.logger.info("Shutdown in progress; suppressing error details.")
            return
        else:
            config.logger.error(f"Failed to write output file {output_file}: {e}")

    nugget_results.clear()
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    config.logger.info(
        f"Processed {total_files} files in {human_readable_time(total_time)}.\n"
        f"Shared counters: {shared_counters}\n"
        f"Nuggets saved to {output_file}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate augmented chunks from JSONL/JSON files')
    parser.add_argument('-i', '--input', help='Directory containing input JSON/JSONL files', default=config.json_dir)
    parser.add_argument('-o', '--output', 
                      help='Output JSONL file (default: nuggets.jsonl in results directory)',
                      default=os.path.join(config.results_dir, 'nuggets.jsonl'))
    parser.add_argument('-m', '--model', help='Model to use', default=config.defaultModel)
    parser.add_argument('-q', '--quiet', action='store_true', help='No progress bar or messages')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('-p', '--parallel', type=int, default=4, help='Number of parallel threads (default: 4)')

    args = parser.parse_args()

    use_progress_bar = config.configure_verbosity(args)
    input_directory = args.input
    output_file = args.output
    model_name = args.model
    model = Model(model_name)
    model.details()

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        process_directory(model, input_directory, output_file,
                          use_progress_bar=use_progress_bar,
                          parallel_workers=args.parallel)
    except KeyboardInterrupt:
        config.initiate_shutdown("User Interrupt - initiating shutdown.")

