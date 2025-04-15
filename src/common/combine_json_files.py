#!/usr/bin/env python

# combine_json_files.py

import json
from pathlib import Path
import argparse
import common.config

# Set up argument parsing
parser = argparse.ArgumentParser(description="Combine MCQ files (JSON and JSONL) into a single JSONL file.")
parser.add_argument('-i', '--input', help='Directory containing input MCQ files', default=common.config.mcq_dir)
parser.add_argument('-o', '--output', help='Output file for combined MCQs (JSONL format)')

args = parser.parse_args()

input_directory = args.input
if args.output:
    output_file = args.output
else:
    # Default output file name if not provided.
    output_file = "MCQ-combined.jsonl"

combined_items = []

# Process both .json and .jsonl files.
for file_path in Path(input_directory).glob("*"):
    if file_path.suffix.lower() not in [".json", ".jsonl"]:
        continue
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().strip()
            if not content:
                print(f"Skipping {file_path}: File is empty.")
                continue
            # If the file is a JSON array (starts with '['), load it as such.
            if content[0] == '[':
                data = json.loads(content)
                if isinstance(data, list):
                    combined_items.extend(data)
                else:
                    print(f"Skipping {file_path}: Not a JSON array.")
            else:
                # Assume JSONL format: one JSON object per line.
                for line in content.splitlines():
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line.strip())
                        # If a line itself contains a list, extend the combined list.
                        if isinstance(item, list):
                            combined_items.extend(item)
                        else:
                            combined_items.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Skipping line in {file_path}: Invalid JSON - {e}")
    except Exception as e:
        print(f"Skipping {file_path}: Error reading file - {e}")

# Sort the combined items (using default large values if keys are missing)
combined_items.sort(key=lambda x: (x.get("filenum", float('inf')), x.get("chunknum", float('inf'))))

# Write output in JSONL format (one JSON object per line)
with open(output_file, "w", encoding="utf-8") as output:
    for item in combined_items:
        output.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Combined JSONL written to {output_file}")

