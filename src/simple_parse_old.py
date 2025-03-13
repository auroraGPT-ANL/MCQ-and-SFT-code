#!/usr/bin/env python

import os
import sys
import json
import re
import PyPDF2
from pdfminer.high_level import extract_text
import config


def clean_string(s: str) -> str:
    """
    Encode to UTF-8 with error handling, then decode back to str.
    - 'replace' will insert a replacement character (�) where invalid surrogates appear.
    - 'ignore' would instead silently remove them.
    """
    return s.encode("utf-8", errors="replace").decode("utf-8")


def clean_data(obj):
    """
    Recursively traverse a Python object (list, dict, string, etc.)
    and clean all strings.
    """
    if isinstance(obj, str):
        return clean_string(obj)
    elif isinstance(obj, dict):
        return {k: clean_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_data(item) for item in obj]
    else:
        return obj


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file, given its file path.
    """
    text_content = []
    with open(pdf_path, 'rb') as f:
        try:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
            return ("\n".join(text_content), 'PyPDF2')
        except:
            config.logger.warning(f'ERROR extracting with PyPDF2 from {str}. Trying pdfminer.')
            text = extract_text(pdf_path)
            return (text, 'pdfminer')


def process_directory(input_dir: str, output_dir: str = "output_files"):
    # Gather all PDF files
    files = [
        f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")
    ]
    total_files = len(files)
    if total_files == 0:
        config.logger.warning("No suitable files found in directory.")
        return

    # Iterate over files
    for i, filename in enumerate(files, start=1):
        file_path = os.path.join(input_dir, filename)

        basename, _ = os.path.splitext(filename)
        out_file = basename + ".json"
        out_path  = os.path.join(output_dir, out_file)
        if os.path.isfile(out_path):
            config.logger.warning(f'Already exists: {i}/{total_files}: {out_path}')
            continue

        config.logger.info(f"Processing file {i}/{total_files}: {file_path}")

        (text, parser) = extract_text_from_pdf(file_path)

        json_structure = {'path': file_path, 'text': text, 'parser':parser}

        with open(out_path, 'w', encoding='utf-8') as out_f:
            json.dump(json_structure, out_f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Program to extract text from PDFs and store in JSON files')
    parser.add_argument('-i', '--input',  help='Directory containing input PDF files',
                        default=config.papers_dir)
    parser.add_argument('-o', '--output', help='Output directory for JSON files',
                        default=config.json_dir)

    args = parser.parse_args()
                                            
    input_directory  = args.input
    output_directory = args.output

    os.makedirs(output_directory, exist_ok=True)
    process_directory(input_directory, output_directory)
