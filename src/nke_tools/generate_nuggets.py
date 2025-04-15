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


def generate_title_author_id(title, first_author):
    """
    Generate a fallback ID from title and author when no DOI or arXiv ID is found.
    
    Args:
        title: The paper title
        first_author: The first author's name
        
    Returns:
        A string in the format "TitleWithoutSpaces-LastName"
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
                response = model.run(
                    system_prompt=config.nugget_prompts['metadata_system'],
                    user_prompt=config.nugget_prompts['metadata_user'].format(chunk=chunk)
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
            response = model.run(
                system_prompt=config.nugget_prompts['metadata_system'],
                user_prompt=config.nugget_prompts['metadata_user'].format(chunk=chunk)
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


def extract_abstract(chunk):
    """Extract and clean the abstract from the first chunk of a paper."""
    try:
        # If chunk is a JSON string, parse it to get the 'text' field
        if isinstance(chunk, str) and chunk.strip().startswith('{'):
            try:
                data = json.loads(chunk)
                chunk = data.get('text', chunk)
            except json.JSONDecodeError:
                pass  # Continue with the original chunk if it's not valid JSON

        # Now look for text between "Abstract" and "I. I"
        abstract_match = re.search(r'Abstract[:\s—-]*(.+?)(?:\\n\\n|\n\n)', chunk, re.DOTALL)
        if abstract_match:
            # Get the abstract text and clean it
            abstract = abstract_match.group(1)
            # Clean up the text:
            # Replace line breaks and multiple spaces with single space
            abstract = re.sub(r'\s+', ' ', abstract)
            # Remove hyphenation artifacts
            abstract = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', abstract)
            # Remove unicode chars
            abstract = abstract.replace('\u201d', '"').replace('\ufb01', 'fi')
            # Final cleanup
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
    # First filter low-confidence facts
    high_confidence_facts = [f for f in facts if f['confidence'] >= confidence_threshold]
    if not high_confidence_facts:
        return []
        
    unique_facts = []
    
    for fact in high_confidence_facts:
        is_duplicate = False
        for existing_fact in unique_facts:
            # Use model to compare facts
            response = model.run(
                system_prompt=config.prompts['fact_comparison_system'],
                user_prompt=config.prompts['fact_comparison_user'].format(
                    fact1=fact['claim'],
                    fact2=existing_fact['claim']
                )
            )
            # Model returns similarity score and reasoning
            try:
                similarity_data = json.loads(response)
                similarity = similarity_data['similarity_score']
                
                if similarity > similarity_threshold:
                    # Keep the fact with higher confidence if they're duplicates
                    if fact['confidence'] > existing_fact['confidence']:
                        unique_facts.remove(existing_fact)
                        is_duplicate = False
                        break
                    else:
                        is_duplicate = True
                        break
            except (json.JSONDecodeError, KeyError) as e:
                config.logger.warning(f"Error parsing similarity response: {e}")
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
        # Format and send the prompt to the model
        formatted_user_message = config.prompts['fact_extraction_user'].format(chunk=chunk)
        response = model.run(
            user_prompt=formatted_user_message,
            system_prompt=config.prompts['fact_extraction_system']
        )
        
        # Parse the response as JSON
        facts = json.loads(response)
        
        # Validate the response format
        if not isinstance(facts, list):
            config.logger.warning(f"Unexpected facts format (not a list): {facts}")
            # Try to handle case where response is wrapped in another object
            if isinstance(facts, dict) and any(k in facts for k in ['facts', 'claims', 'results']):
                for key in ['facts', 'claims', 'results']:
                    if key in facts and isinstance(facts[key], list):
                        facts = facts[key]
                        break
            else:
                # If we can't salvage the response, return an empty list
                return []
                
        # Ensure each fact has the required keys
        validated_facts = []
        for fact in facts:
            if isinstance(fact, dict) and 'claim' in fact and 'span' in fact and 'confidence' in fact:
                # Ensure confidence is a float between 0 and 1
                try:
                    fact['confidence'] = float(fact['confidence'])
                    if not (0 <= fact['confidence'] <= 1):
                        fact['confidence'] = max(0, min(1, fact['confidence']))
                except (ValueError, TypeError):
                    fact['confidence'] = 0.5  # Default if invalid
                    
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
                  pbar_total, pbar_success, shared_counters, counter_lock, 
                  doi=None, is_first_chunk=False):
    """
    Process a single chunk to extract atomic facts.
    For the first chunk, extract the abstract instead of extracting facts.
    Returns a tuple: (filename, linenum, chunknum, result_dict or None, success_flag)
    """
    chunk_success = False
    nugget = None

    if config.shutdown_event.is_set():
        config.logger.info(f"Shutting down: Skipping chunk {chunknum} in file {filename}.")
        return (filename, linenum, chunknum, None, False)

    config.logger.info(f"Processing chunk {chunknum} in file {filename}.")

    # Extract atomic facts from chunk
    try:
        # For first chunk, try to extract abstract
        if is_first_chunk:
            try:
                abstract = extract_abstract(chunk)
                if abstract:
                    nugget = {
                        "doi": doi,
                        "abstract": abstract,
                        "facts": []  # Empty facts list for abstract
                    }
                    chunk_success = True
                    config.logger.info("Successfully extracted abstract for first chunk")
                else:
                    config.logger.warning("Could not extract abstract, falling back to normal processing")
            except Exception as e:
                config.logger.warning(f"Error extracting abstract: {e}, falling back to normal processing")
                
        # If not first chunk or abstract extraction failed, extract atomic facts
        if not is_first_chunk or not chunk_success:
            # Extract atomic facts
            facts = extract_atomic_facts(chunk, model)
            
            # Create the nugget with extracted facts
            nugget = {
                "doi": doi,
                "facts": facts
            }
            chunk_success = True
            
            # Only attempt local (chunk-level) deduplication if we have facts
            if facts and len(facts) > 1:
                facts = deduplicate_facts(facts, model, confidence_threshold=0.0)  # Don't filter by confidence yet
                config.logger.info(f"Deduplicated to {len(facts)} unique facts within chunk")
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


def write_nuggets_to_file(nuggets, output_file, append=True):
    """
    Write a list of nuggets to the output file.
    
    Args:
        nuggets (list): List of nugget dictionaries to write
        output_file (str): Path to the output file
        append (bool): Whether to append to the file (True) or overwrite (False)
    
    Returns:
        bool: True if successful, False if an error occurred
    """
    if not nuggets:
        return True  # Nothing to write
        
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:  # If output_file includes a directory path
        os.makedirs(output_dir, exist_ok=True)
    
    # Use atomic file operations with locking for thread safety
    with config.output_file_lock:
        try:
            mode = 'a' if append else 'w'
            config.logger.info(f"Writing {len(nuggets)} nuggets to {output_file}")
            with open(output_file, mode, encoding='utf-8') as out_f:
                for nugget in nuggets:
                    # Ensure facts are present (even if empty) in the output
                    if 'facts' not in nugget and 'augmented_chunk' in nugget:
                        # Handle legacy format
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

    counter_lock = threading.Lock()
    shared_counters = {"success": 0, "failure": 0}
    # Cache for paper DOIs and titles
    paper_metadata = {}
    
    # Counter for total nuggets processed
    total_nuggets_processed = 0
    
    # Track all facts for global deduplication
    all_facts = []

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
                        first_line = lines[0].strip()
                        try:
                            first_record = json.loads(first_line)
                            first_text = first_record.get('text', '')
                        except json.JSONDecodeError:
                            config.logger.warning(f"Error parsing JSON from first line of {filename}")
                            first_text = first_line  # Use the raw line if it's not valid JSON
                        if first_text:
                            first_chunks = split_text_into_chunks(first_text, CHUNK_SIZE)
                            if first_chunks:
                                try:
                                    config.logger.info(f"Processing first chunk of {filename} for metadata")
                                    metadata = extract_paper_metadata(first_chunks[0], model)
                                    
                                    # Check for DOI in metadata identifiers
                                    identifiers = metadata.get('identifiers', [])
                                    
                                    # First check for DOI in identifiers
                                    if identifiers and any(id.startswith('10.') for id in identifiers):
                                        doi = next((id for id in identifiers if id.startswith('10.')), None)
                                        paper_metadata[filename] = {
                                            'doi': doi,
                                            'title': metadata.get('title'),
                                            'first_author': metadata.get('first_author')
                                        }
                                        config.logger.info(f"Found DOI in paper: {doi}")
                                    
                                    # If DOI not found, check for arXiv ID
                                    elif identifiers and any('arxiv' in id.lower() for id in identifiers):
                                        arxiv_id = next((id for id in identifiers if 'arxiv' in id.lower()), None)
                                        paper_metadata[filename] = {
                                            'doi': arxiv_id,  # Use arXiv ID as the identifier
                                            'title': metadata.get('title'),
                                            'first_author': metadata.get('first_author')
                                        }
                                        config.logger.info(f"Using arXiv ID as identifier: {arxiv_id}")
                                    
                                    # If no identifiers found, try DOI lookup
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
                                            # Generate fallback ID from title and author
                                            fallback_id = generate_title_author_id(metadata['title'], metadata['first_author'])
                                            paper_metadata[filename] = {
                                                'doi': fallback_id,
                                                'title': metadata['title'],
                                                'first_author': metadata['first_author']
                                            }
                                            config.logger.info(f"Using fallback title-author ID for {filename}: {fallback_id}")
                                except Exception as e:
                                    config.logger.warning(f"Error processing first chunk of {filename} for metadata: {e}")
                    except Exception as e:
                        config.logger.warning(f"Error parsing JSON for {filename}: {e}")
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

            # Process each file completely, one at a time
            file_futures = []
            file_nuggets = []
            
            # Get metadata for this file
            file_doi = None
            if filename in paper_metadata:
                file_doi = paper_metadata[filename].get('doi')
                
            # Process all chunks from the current file
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
                        
                    # Pass is_first_chunk=True for the first chunk of each file
                    future = executor.submit(process_chunk, model, filename, rec_path,
                                             linenum, chunknum, chunk,
                                             pbar_total, pbar_success,
                                             shared_counters, counter_lock,
                                             file_doi,
                                             is_first_chunk=(linenum==1 and chunknum==1))
                    file_futures.append(future)
                    
            # Wait for all chunks from this file to complete
            processed_chunks = set()  # Track processed chunks for this file
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
            
            # Write nuggets for this file immediately
            # Write nuggets for this file immediately
            if file_nuggets:
                # Perform global deduplication across all facts
                all_file_facts = []
                for nugget in file_nuggets:
                    all_file_facts.extend(nugget.get('facts', []))
                
                if all_file_facts:
                    # Get the current DOI
                    current_doi = file_nuggets[0].get('doi')
                    
                    # Apply confidence filtering and local deduplication
                    local_unique_facts = deduplicate_facts(
                        all_file_facts, 
                        model, 
                        similarity_threshold=0.88,
                        confidence_threshold=0.5
                    )
                    
                    config.logger.info(f"File {filename}: {len(all_file_facts)} facts -> {len(local_unique_facts)} unique facts after local deduplication")
                    
                    # If we have previous facts, perform global deduplication
                    if all_facts:
                        # Check new facts against all previous facts
                        combined_facts = local_unique_facts + all_facts
                        global_unique_facts = deduplicate_facts(
                            combined_facts,
                            model,
                            similarity_threshold=0.88,
                            confidence_threshold=0.5
                        )
                        
                        # Identify truly new facts (not in all_facts)
                        new_facts = []
                        for fact in global_unique_facts:
                            # Check if this fact is from the current file (not in all_facts)
                            is_new = True
                            for old_fact in all_facts:
                                if fact['claim'] == old_fact['claim'] and fact['span'] == old_fact['span']:
                                    is_new = False
                                    break
                            if is_new:
                                new_facts.append(fact)
                        
                        config.logger.info(f"File {filename}: Found {len(new_facts)} new facts after global deduplication")
                        
                        # Update the global facts collection
                        all_facts = global_unique_facts
                        
                        # Create nugget with just the new facts from this file
                        if new_facts:
                            file_nugget = {
                                'doi': current_doi,
                                'facts': new_facts
                            }
                            write_nuggets_to_file([file_nugget], output_file)
                            total_nuggets_processed += 1
                    else:
                        # First file being processed - all facts are new
                        all_facts = local_unique_facts
                        
                        # Create nugget with all unique facts from this file
                        file_nugget = {
                            'doi': current_doi,
                            'facts': local_unique_facts
                        }
                        write_nuggets_to_file([file_nugget], output_file)
                        total_nuggets_processed += 1
                else:
                    # No facts in this file, just write the original nuggets
                    write_nuggets_to_file(file_nuggets, output_file)
                    total_nuggets_processed += len(file_nuggets)
                    
                file_nuggets.clear()  # Free memory
                
            # Clear futures for this file
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

