#!/usr/bin/env python
"""
simple_parse.py

Convert PDF to JSON. Provides both a CLI entrypoint and a Python API.
"""
import os
import sys
import json
import argparse
import PyPDF2
from pdfminer.high_level import extract_text
from common import config
from tqdm import tqdm


def clean_string(s: str) -> str:
    """
    Encode to UTF-8 with error handling, then decode back to str.
    """
    return s.encode("utf-8", errors="replace").decode("utf-8")


def clean_data(obj):
    """
    Recursively traverse a Python object and clean all strings.
    """
    if isinstance(obj, str):
        return clean_string(obj)
    elif isinstance(obj, dict):
        return {k: clean_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_data(item) for item in obj]
    else:
        return obj


def extract_text_from_pdf(pdf_path: str) -> (str, str):
    """
    Extract all text from a PDF file, returning (text, parser_used).
    Returns (None, None) if both parsers fail.
    """
    text_content = []
    with open(pdf_path, 'rb') as f:
        # First try PyPDF2
        try:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
            return ("\n".join(text_content), 'PyPDF2')
        except Exception:
            config.logger.warning(f'ERROR extracting with PyPDF2 from {pdf_path}. Trying pdfminer.')
            # Now try pdfminer
            try:
                text = extract_text(pdf_path)
                return (text, 'pdfminer')
            except Exception as e:
                # Give INFO and skip this file
                config.logger.info(f'ERROR extracting with pdfminer from {pdf_path}: {e}. Skipping file.')
                return (None, None)


def process_directory(input_dir: str, output_dir: str = "output_files", use_progress_bar: bool = True):
    """
    Iterate over PDFs in input_dir, extract text, and write JSON files to output_dir.
    """
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    total_files = len(files)
    if total_files == 0:
        config.logger.warning(f"No PDF papers found in {input_dir}.")
        return

    if use_progress_bar:
        pbar = tqdm(total=total_files, desc="Processing PDFs", unit="file")
    else:
        pbar = config.NoOpTqdm(total=total_files)

    os.makedirs(output_dir, exist_ok=True)

    for i, filename in enumerate(files, start=1):
        file_path = os.path.join(input_dir, filename)
        basename, _ = os.path.splitext(filename)
        out_path = os.path.join(output_dir, basename + ".json")

        if os.path.isfile(out_path):
            config.logger.info(f'Already exists: {i}/{total_files}: {out_path}')
            pbar.update(1)
            continue

        config.logger.info(f"Processing file {i}/{total_files}: {file_path}")
        text, parser = extract_text_from_pdf(file_path)
        if text is None:
            # Skipped because both parsers failed
            pbar.update(1)
            continue

        json_structure = {'path': file_path, 'text': text, 'parser': parser}
        with open(out_path, 'w', encoding='utf-8') as out_f:
            json.dump(json_structure, out_f, ensure_ascii=False, indent=2)
        pbar.update(1)

    pbar.close()
    config.logger.info("Processing complete")


def parse_pdfs_dir(input_dir: str,
                   output_dir: str,
                   use_progress_bar: bool = True) -> str:
    """
    API function: extract all PDFs from input_dir, write JSON to output_dir,
    and return the output_dir path.
    """
    os.makedirs(output_dir, exist_ok=True)
    process_directory(input_dir, output_dir, use_progress_bar)
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Convert PDF files to JSON via simple_parse.py'
    )
    parser.add_argument('-i', '--input', help='Directory containing input PDF files',
                        default=config.papers_dir)
    parser.add_argument('-o', '--output', help='Output directory for JSON files',
                        default=config.json_dir)
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='No progress bar or messages')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()
    use_pb = config.configure_verbosity(args)

    result_dir = parse_pdfs_dir(
        input_dir=args.input,
        output_dir=args.output,
        use_progress_bar=use_pb
    )
    # Print result for CLI capture
    print(result_dir)


if __name__ == "__main__":
    main()

