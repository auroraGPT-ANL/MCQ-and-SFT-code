#!/usr/bin/env python
"""
select_mcqs_at_random.py

Select N random entries from a JSON or JSONL file and write them to another JSONL file.
Provides a Python API and CLI entrypoint.
"""
import json
import random
import argparse
from typing import List


def select_random_entries(input_file: str, output_file: str, n: int) -> str:
    """
    Selects N random entries from a JSON or JSONL file and writes them to output_file.

    Args:
      input_file: Path to the input JSON (array) or JSONL file.
      output_file: Path to the output JSONL file.
      n: Number of random entries to select.

    Returns:
      The path to the output_file.
    """
    # Load the JSON data (either array or JSONL)
    data: List[dict] = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        first_char = infile.read(1)
        infile.seek(0)
        if first_char == '[':
            data = json.load(infile)
        else:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))

    if n > len(data):
        raise ValueError(f"Requested {n} entries, but the file only contains {len(data)} entries.")

    selected_entries = random.sample(data, n)

    # Write out as JSONL
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in selected_entries:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Select N random entries from a JSON or JSONL file and write to JSONL"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to the input JSON or JSONL file"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Path to the output JSONL file"
    )
    parser.add_argument(
        "-n", "--number", required=True, type=int,
        help="Number of random entries to select"
    )

    args = parser.parse_args()

    try:
        out_path = select_random_entries(args.input, args.output, args.number)
        print(out_path)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()

