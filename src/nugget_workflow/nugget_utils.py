#!/usr/bin/env python

# generate_nuggets.py

import os
import sys
import json
import re
import time
from openai import OpenAI
import spacy
import argparse
import logging
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import TimeoutError
import threading

from common import config  # Keep for backward compatibility
from common.loader import load_settings
from common.model_access import Model

# Initialize settings
settings = load_settings()

##############################################################################
# Global constants
##############################################################################
# Use settings with fallback to config for backward compatibility
CHUNK_SIZE = settings.quality.chunkSize if hasattr(settings, 'quality') else config.chunkSize

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
                #config.logger.info("Shutdown in progress; suppressing error details.")
                return 0
            else:
                config.logger.info(f"Failed to read file {path}: {e}")
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


def generate_title_author_id(title, first_author):
    """
    Generate a fallback ID from title and author when no DOI or arXiv ID is found.
    """
    if not title or not first_author:
        return None

    try:
        # Remove spaces from title
        title_part = title.replace(" ", "")

        # Extract last name of first author
        author_parts = first_author.split()
        if len(author_parts) > 0:
            last_name = author_parts[-1]  # Assume last part is the last name

            # Create the fallback ID
            fallback_id = f"{title_part}-{last_name}"
            config.logger.info(f"Generated title-author ID: {fallback_id}")
            return fallback_id
    except Exception as e:
        config.logger.warning(f"Error generating title-author ID: {e}")

    return None


def extract_paper_metadata(chunk, model):
    """Extract paper metadata including arXiv ID if available."""
    try:
        # Initialize metadata with empty values
        metadata = {
            "identifiers": [],
            "title": None,
            "first_author": None
        }

        # First check for DOI in paper
        doi_match = re.search(r'(?:doi|DOI|https://doi\.org)/(\S+)', chunk)
        if doi_match:
            doi = doi_match.group(1)
            config.logger.info(f"Found DOI in paper: {doi}")
            metadata["identifiers"] = [f"10.{doi}" if not doi.startswith('10.') else doi]

            # We still need to try to get the title and author for fallback purposes
            try:
                # Get prompts from settings if available, otherwise fall back to config
                metadata_system = settings.nugget_prompts.get('metadata_system') if hasattr(settings, 'nugget_prompts') else config.nugget_prompts['metadata_system']
                metadata_user = settings.nugget_prompts.get('metadata_user') if hasattr(settings, 'nugget_prompts') else config.nugget_prompts['metadata_user']

                response = model.run(
                    system_prompt=metadata_system,
                    user_prompt=metadata_user.format(chunk=chunk)
                )
                model_metadata = json.loads(response)
                metadata["title"] = model_metadata.get("title")
                metadata["first_author"] = model_metadata.get("first_author")
            except Exception as e:
                config.logger.warning(f"Error extracting title/author after finding DOI: {e}")

            return metadata

        # If no DOI, check for arXiv ID
        arxiv_match = re.search(r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)', chunk)
        if arxiv_match:
            arxiv_id = arxiv_match.group(0)  # Full "arXiv:XXXX.XXXXX" format
            config.logger.info(f"Found arXiv ID in text: {arxiv_id}")
            metadata["identifiers"] = [arxiv_id]

            # We still need to try to get the title and author for fallback purposes
            try:
                response = model.run(
                    system_prompt=config.nugget_prompts['metadata_system'],
                    user_prompt=config.nugget_prompts['metadata_user'].format(chunk=chunk)
                )
                model_metadata = json.loads(response)
                metadata["title"] = model_metadata.get("title")
                metadata["first_author"] = model_metadata.get("first_author")
            except Exception as e:
                config.logger.warning(f"Error extracting title/author after finding arXiv ID: {e}")

            return metadata

        # If no direct matches, use the model to extract metadata
        try:
            # Get prompts from settings if available, otherwise fall back to config
            metadata_system = settings.nugget_prompts.get('metadata_system') if hasattr(settings, 'nugget_prompts') else config.nugget_prompts['metadata_system']
            metadata_user = settings.nugget_prompts.get('metadata_user') if hasattr(settings, 'nugget_prompts') else config.nugget_prompts['metadata_user']

            response = model.run(
                system_prompt=metadata_system,
                user_prompt=metadata_user.format(chunk=chunk)
            )

            model_metadata = json.loads(response)
            metadata["title"] = model_metadata.get("title")
            metadata["first_author"] = model_metadata.get("first_author")

            # Double-check the response for arXiv IDs or DOIs
            identifiers = model_metadata.get('identifiers', [])
            if identifiers:
                arxiv_id = next((id for id in identifiers if 'arxiv' in id.lower()), None)
                if arxiv_id:
                    config.logger.info(f"Model identified arXiv ID: {arxiv_id}")
                    metadata["identifiers"] = [arxiv_id]
                else:
                    doi_id = next((id for id in identifiers if id.startswith('10.')), None)
                    if doi_id:
                        config.logger.info(f"Model identified DOI: {doi_id}")
                        metadata["identifiers"] = [doi_id]

            # If no DOI or arXiv ID, generate a fallback ID from title and author
            if not metadata["identifiers"] and metadata["title"] and metadata["first_author"]:
                fallback_id = generate_title_author_id(metadata["title"], metadata["first_author"])
                if fallback_id:
                    metadata["identifiers"] = [fallback_id]
                    config.logger.info(f"Using fallback title-author ID: {fallback_id}")

            return metadata
        except Exception as e:
            config.logger.warning(f"Error using model to extract metadata: {e}")
            return metadata
    except Exception as e:
        config.logger.warning(f"Error extracting paper metadata: {e}")
        return metadata

def lookup_doi(title, first_author, model):
    """Search for and return the paper's DOI using title and first author."""
    try:
        # Get prompts from settings if available, otherwise fall back to config
        doi_system = settings.nugget_prompts.get('doi_system') if hasattr(settings, 'nugget_prompts') else config.nugget_prompts['doi_system']
        doi_user = settings.nugget_prompts.get('doi_user') if hasattr(settings, 'nugget_prompts') else config.nugget_prompts['doi_user']

        response = model.run(
            system_prompt=doi_system,
            user_prompt=doi_user.format(
                title=title,
                first_author=first_author
            )
        )

        doi = response.strip().strip('"').strip("'")
        if doi.lower() == 'null':
            return None
        if not doi.startswith('10.'):
            return None
        return doi
    except Exception as e:
        config.logger.warning(f"Error looking up DOI: {e}")
        return None


def extract_abstract(chunk):
    """Extract and clean the abstract from the first chunk of a paper."""
    try:
        if isinstance(chunk, str) and chunk.strip().startswith('{'):
            try:
                data = json.loads(chunk)
                chunk = data.get('text', chunk)
            except json.JSONDecodeError:
                pass

        abstract_match = re.search(r'Abstract[:\s—-]*(.+?)(?:\\n\\n|\n\n)', chunk, re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1)
            abstract = re.sub(r'\s+', ' ', abstract)
            abstract = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', abstract)
            abstract = abstract.replace('\u201d', '"').replace('\ufb01', 'fi')
            abstract = abstract.strip()
            return abstract
        return None
    except Exception as e:
        config.logger.warning(f"Error extracting abstract: {e}")
        return None


def deduplicate_facts(facts, model, similarity_threshold=0.88, confidence_threshold=0.5):
    """
    Cluster similar facts and filter low-confidence ones.

    Args:
        facts (list): List of fact dictionaries with 'claim' keys
        model (Model): The model instance to use for similarity comparison
        similarity_threshold (float): Threshold above which facts are considered duplicates
        confidence_threshold (float): Minimum confidence score to keep a fact

    Returns:
        list: Filtered and de-duplicated list of facts
    """
    if not isinstance(facts, list):
        config.logger.warning(f"Expected list of facts, got {type(facts)}")
        return []

    # Ensure all facts have valid confidence values and are properly structured
    validated_facts = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue

        if 'claim' not in fact or 'confidence' not in fact:
            continue

        # Convert confidence to float if it's not already
        try:
            fact['confidence'] = float(fact['confidence'])
        except (ValueError, TypeError):
            fact['confidence'] = 0.5  # Default if conversion fails

        validated_facts.append(fact)

    # Filter by confidence
    high_confidence_facts = [f for f in validated_facts if f['confidence'] >= confidence_threshold]
    if not high_confidence_facts:
        return []

    unique_facts = []

    for fact in high_confidence_facts:
        is_duplicate = False
        for existing_fact in unique_facts:
            try:
                # Get prompts from settings if available, otherwise fall back to config
                fact_comparison_system = settings.prompts.get('fact_comparison_system') if hasattr(settings, 'prompts') else config.prompts['fact_comparison_system']
                fact_comparison_user = settings.prompts.get('fact_comparison_user') if hasattr(settings, 'prompts') else config.prompts['fact_comparison_user']

                response = model.run(
                    system_prompt=fact_comparison_system,
                    user_prompt=fact_comparison_user.format(
                        fact1=fact['claim'],
                        fact2=existing_fact['claim']
                    )
                )

                similarity_data = json.loads(response)
                similarity = float(similarity_data.get('similarity_score', 0))

                if similarity > similarity_threshold:
                    if fact['confidence'] > existing_fact['confidence']:
                        unique_facts.remove(existing_fact)
                        is_duplicate = False
                        break
                    else:
                        is_duplicate = True
                        break
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                config.logger.warning(f"Error comparing facts: {e}")
                continue

        if not is_duplicate:
            unique_facts.append(fact)

    return unique_facts


def extract_atomic_facts(chunk, model):
    """
    Extract atomic factual statements from a chunk of text.

    Args:
        chunk (str): The text chunk to extract facts from
        model (Model): The model instance to use for extraction

    Returns:
        list: A list of dicts with structure: [{"claim": "...", "span": "...", "confidence": 0.95}]
    """
    try:
        # Get prompts from settings if available, otherwise fall back to config
        fact_extraction_user = settings.prompts.get('fact_extraction_user') if hasattr(settings, 'prompts') else config.prompts['fact_extraction_user']
        fact_extraction_system = settings.prompts.get('fact_extraction_system') if hasattr(settings, 'prompts') else config.prompts['fact_extraction_system']

        formatted_user_message = fact_extraction_user.format(chunk=chunk)
        response = model.run(
            user_prompt=formatted_user_message,
            system_prompt=fact_extraction_system
        )

        facts = json.loads(response)

        if not isinstance(facts, list):
            config.logger.warning(f"Unexpected facts format (not a list): {facts}")
            if isinstance(facts, dict) and any(k in facts for k in ['facts', 'claims', 'results']):
                for key in ['facts', 'claims', 'results']:
                    if key in facts and isinstance(facts[key], list):
                        facts = facts[key]
                        break
            else:
                return []

        validated_facts = []
        for fact in facts:
            if isinstance(fact, dict) and 'claim' in fact and 'span' in fact and 'confidence' in fact:
                try:
                    fact['confidence'] = float(fact['confidence'])
                    if not (0 <= fact['confidence'] <= 1):
                        fact['confidence'] = max(0, min(1, fact['confidence']))
                except (ValueError, TypeError):
                    fact['confidence'] = 0.5
                validated_facts.append(fact)
            else:
                config.logger.warning(f"Skipping fact with missing required keys: {fact}")

        config.logger.info(f"Extracted {len(validated_facts)} valid facts from chunk")
        return validated_facts

    except json.JSONDecodeError as e:
        config.logger.warning(f"Failed to parse model response as JSON: {e}")
        return []
    except Exception as e:
        config.logger.warning(f"Error extracting atomic facts: {e}")
        return []


def format_content(response):
    """Helper function to consistently format nugget content."""
    try:
        if isinstance(response, str):
            response = response.replace('augmented_chunk:', '').strip()
            content = json.loads(response.replace("'", '"'))
        else:
            content = response

        formatted_parts = []

        if isinstance(content, dict):
            if 'augmented_chunk' in content:
                content = content['augmented_chunk']

            if isinstance(content, dict):
                points = []
                for key in ['bullet_points', 'summary_points', 'summary']:
                    if key in content:
                        for point in content[key]:
                            cleaned_point = re.sub(r'^\d+\.\s*', '', point.strip())
                            points.append(cleaned_point)
                        break

                if points:
                    formatted_parts.extend(f"• {point}" for point in points)

                comments = content.get('comments', content.get('Expansion', ''))
                if comments:
                    if formatted_parts:
                        formatted_parts.append("")
                    formatted_parts.append("Expanded Context:")
                    formatted_parts.append(comments.strip())

        if formatted_parts:
            return "\n".join(formatted_parts)
        else:
            return str(content).strip()

    except json.JSONDecodeError:
        return response.replace('augmented_chunk:', '').strip()


def process_chunk(model, filename, file_path, linenum, chunknum, chunk,
                  pbar_total, pbar_success, shared_counters, counter_lock,
                  doi=None, is_first_chunk=False):
    """
    Process a single chunk to extract atomic facts.
    For the first chunk, attempt to extract the abstract; if unsuccessful or if not first chunk,
    extract atomic facts local to the chunk.
    Returns a tuple: (filename, linenum, chunknum, nugget dict or None, success_flag)
    """
    chunk_success = False
    nugget = None

    if config.shutdown_event.is_set():
        config.logger.info(f"Shutting down: Skipping chunk {chunknum} in file {filename}.")
        return (filename, linenum, chunknum, None, False)

    config.logger.info(f"Processing chunk {chunknum} in file {filename}.")

    try:
        # Attempt abstract extraction only for the first chunk.
        if is_first_chunk:
            try:
                abstract = extract_abstract(chunk)
                if abstract:
                    nugget = {
                        "doi": doi,
                        "abstract": abstract,
                        "facts": []  # No facts for abstract nugget
                    }
                    chunk_success = True
                    config.logger.info("Successfully extracted abstract for first chunk.")
                else:
                    config.logger.warning("Abstract extraction returned no result; falling back to fact extraction.")
            except Exception as e:
                config.logger.warning(f"Error extracting abstract: {e}, falling back to fact extraction.")

        # If the nugget was not already created (or not the first chunk), extract facts.
        if nugget is None:
            facts = extract_atomic_facts(chunk, model)
            if isinstance(facts, list) and facts:
                nugget = {
                    "doi": doi,
                    "facts": facts  # Use facts directly since they're already validated in extract_atomic_facts
                }
                chunk_success = True
                config.logger.info(f"Successfully processed {len(facts)} facts")
            else:
                config.logger.warning(f"No valid facts extracted for chunk {chunknum}")

    except Exception as e:
        if config.shutdown_event.is_set():
            config.logger.info("Shutdown in progress; suppressing error details.")
            return (filename, linenum, chunknum, None, False)
        else:
            config.logger.warning(f"Error processing chunk {chunknum} in file {filename}: {e}")
            pbar_total.update(1)
            update_shared_counters(False, shared_counters, counter_lock)
            return (filename, linenum, chunknum, None, False)

    pbar_total.update(1)
    update_shared_counters(chunk_success, shared_counters, counter_lock)
    if chunk_success:
        pbar_success.update(1)
    return (filename, linenum, chunknum, nugget, chunk_success)


def write_nuggets_to_file(nuggets, output_file, append=True):
    """
    Write a list of nuggets to the output file.
    """
    if not nuggets:
        return True

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with config.output_file_lock:
        try:
            mode = 'a' if append else 'w'
            config.logger.info(f"Writing {len(nuggets)} nuggets to {output_file}")
            with open(output_file, mode, encoding='utf-8') as out_f:
                for nugget in nuggets:
                    if 'facts' not in nugget and 'augmented_chunk' in nugget:
                        nugget['facts'] = []
                    out_f.write(json.dumps(nugget, ensure_ascii=False) + "\n")
            config.logger.info(f"Successfully wrote {len(nuggets)} nuggets to {output_file}")
            return True
        except Exception as e:
            if config.shutdown_event.is_set():
                config.logger.info("Shutdown in progress; suppressing error details.")
            else:
                config.logger.error(f"Failed to write output file {output_file}: {e}")
            return False


def process_directory(model, input_dir: str, output_file: str = "nuggets.jsonl",
                      use_progress_bar: bool = True, parallel_workers: int = 4):
    """
    Process all JSON/JSONL files by scheduling each chunk as a separate task.
    Nuggets are written to the output file incrementally after each file is processed.
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

    # Initialize a dictionary for paper metadata.
    paper_metadata = {}

    counter_lock = threading.Lock()
    shared_counters = {"success": 0, "failure": 0}
    total_nuggets_processed = 0

    # Removed the duplicate "all_facts = []" block

    if use_progress_bar:
        pbar_total = tqdm(total=approximate_chunk_count, desc=" Processed", position=0, unit="chunk")
        pbar_success = tqdm(total=approximate_chunk_count, desc="Successful", position=1, unit="chunk")
    else:
        pbar_total = config.NoOpTqdm()
        pbar_success = config.NoOpTqdm()

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        # First pass: Process first chunk of each file for metadata extraction.
        for filename in all_files:
            file_path = os.path.join(input_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    if filename.lower().endswith(".json"):
                        json_str = file.read()
                        lines = [json_str]
                    else:
                        lines = file.readlines()

                    try:
                        first_line = lines[0].strip()
                        try:
                            first_record = json.loads(first_line)
                            first_text = first_record.get('text', '')
                        except json.JSONDecodeError:
                            config.logger.warning(f"Error parsing JSON from first line of {filename}")
                            first_text = first_line
                        if first_text:
                            first_chunks = split_text_into_chunks(first_text, CHUNK_SIZE)
                            if first_chunks:
                                config.logger.info(f"Processing first chunk of {filename} for metadata")
                                metadata = extract_paper_metadata(first_chunks[0], model)
                                identifiers = metadata.get('identifiers', [])

                                if identifiers and any(id.startswith('10.') for id in identifiers):
                                    doi = next((id for id in identifiers if id.startswith('10.')), None)
                                    paper_metadata[filename] = {
                                        'doi': doi,
                                        'title': metadata.get('title'),
                                        'first_author': metadata.get('first_author')
                                    }
                                    config.logger.info(f"Found DOI in paper: {doi}")
                                elif identifiers and any('arxiv' in id.lower() for id in identifiers):
                                    arxiv_id = next((id for id in identifiers if 'arxiv' in id.lower()), None)
                                    paper_metadata[filename] = {
                                        'doi': arxiv_id,
                                        'title': metadata.get('title'),
                                        'first_author': metadata.get('first_author')
                                    }
                                    config.logger.info(f"Using arXiv ID as identifier: {arxiv_id}")
                                elif metadata.get('title') and metadata.get('first_author'):
                                    doi = lookup_doi(metadata['title'], metadata['first_author'], model)
                                    if doi:
                                        paper_metadata[filename] = {
                                            'doi': doi,
                                            'title': metadata['title'],
                                            'first_author': metadata.get('first_author')
                                        }
                                        config.logger.info(f"Found DOI for {filename}: {doi}")
                                    else:
                                        fallback_id = generate_title_author_id(metadata['title'], metadata['first_author'])
                                        paper_metadata[filename] = {
                                            'doi': fallback_id,
                                            'title': metadata['title'],
                                            'first_author': metadata['first_author']
                                        }
                                        config.logger.info(f"Using fallback title-author ID for {filename}: {fallback_id}")
                                else:
                                    config.logger.warning(f"No identifiers found for {filename}.")
                    except Exception as e:
                        config.logger.warning(f"Error parsing JSON for {filename}: {e}")
            except Exception as e:
                config.logger.error(f"Failed to read file {filename} for metadata extraction: {e}")

        # Second pass: Process each file completely.
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

            file_futures = []
            file_nuggets = []

            file_doi = None
            if filename in paper_metadata:
                file_doi = paper_metadata[filename].get('doi')

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
                                             shared_counters, counter_lock,
                                             file_doi,
                                             is_first_chunk=(linenum==1 and chunknum==1))
                    file_futures.append(future)

            processed_chunks = set()
            for future in concurrent.futures.as_completed(file_futures):
                try:
                    fname, linenum, chunknum, nugget, success = future.result(timeout=75)
                    if nugget is not None:
                        chunk_id = (linenum, chunknum)
                        if chunk_id not in processed_chunks:
                            file_nuggets.append(nugget)
                            processed_chunks.add(chunk_id)
                except TimeoutError:
                    config.logger.warning(f"Chunk processing task for {filename} timed out after 75s")
                except Exception as e:
                    if config.shutdown_event.is_set():
                        config.logger.info("Shutdown in progress; suppressing error details.")
                        continue
                    else:
                        config.logger.error(f"Error processing a chunk in {filename}: {e}")

            if file_nuggets:
                # Write nuggets for each file immediately without file-level deduplication.
                write_nuggets_to_file(file_nuggets, output_file)
                total_nuggets_processed += len(file_nuggets)
                file_nuggets.clear()  # Clear once to free memory
            file_futures.clear()
    if use_progress_bar:
        remaining = pbar_total.total - pbar_total.n
        if remaining > 0:
            pbar_total.update(remaining)
        pbar_total.close()
        pbar_success.close()

    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    config.logger.info(
        f"Processed {total_files} files in {human_readable_time(total_time)}.\n"
        f"Shared counters: {shared_counters}\n"
        f"Total nuggets saved to {output_file}: {total_nuggets_processed}"
    )


