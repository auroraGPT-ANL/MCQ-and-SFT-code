#!/usr/bin/env python
"""
combine_json_files.py

Combine MCQ files (JSON and JSONL) into a single JSONL file.
Provides both a CLI entrypoint and a Python API function.
"""
import json
import argparse
from pathlib import Path
from common import config


def combine_json_to_jsonl(input_dir: str, output_file: str = None) -> str:
    """
    Combine all .json and .jsonl files in input_dir into one JSONL file.

    Args:
        input_dir: Directory containing input MCQ files.
        output_file: Path for the combined JSONL output. If None,
                     defaults to 'MCQ-combined.jsonl' in cwd or config.mcq_dir.

    Returns:
        The path to the combined JSONL file.
    """
    # Determine default output path
    if not output_file:
        output_file = "MCQ-combined.jsonl"
    combined_items = []

    # Process both .json and .jsonl files
    for file_path in Path(input_dir).glob("*"):
        suffix = file_path.suffix.lower()
        if suffix not in [".json", ".jsonl"]:
            continue
        try:
            content = file_path.read_text(encoding='utf-8').strip()
            if not content:
                print(f"Skipping {file_path}: File is empty.")
                continue
            # JSON array
            if content[0] == '[':
                data = json.loads(content)
                if isinstance(data, list):
                    combined_items.extend(data)
                else:
                    print(f"Skipping {file_path}: Not a JSON array.")
            else:
                # JSONL: one JSON object or list per line
                for line in content.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if isinstance(item, list):
                            combined_items.extend(item)
                        else:
                            combined_items.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Skipping line in {file_path}: Invalid JSON - {e}")
        except Exception as e:
            print(f"Skipping {file_path}: Error reading file - {e}")

    # Sort combined items by filenum, chunknum
    combined_items.sort(key=lambda x: (x.get('filenum', float('inf')),
                                       x.get('chunknum', float('inf'))))

    # Write output in JSONL format
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for item in combined_items:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Combine MCQ files into a single JSONL file"
    )
    parser.add_argument('-i', '--input',
                        help='Directory containing input MCQ files',
                        default=config.mcq_dir)
    parser.add_argument('-o', '--output',
                        help='Output file for combined MCQs (JSONL format)',
                        default=None)

    args = parser.parse_args()
    output_path = combine_json_to_jsonl(args.input, args.output)
    print(f"Combined JSONL written to {output_path}")


if __name__ == '__main__':
    main()

