#!/usr/bin/env python

import json
import random
import argparse

def select_random_entries(input_file, output_file, n):
    """
    Selects N random entries from a JSON file and writes them to another JSON file.

    Args:
    - input_file (str): Path to the input JSON file.
    - output_file (str): Path to the output JSON file.
    - n (int): Number of random entries to select.
    """
    # Load the JSON data
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    # Ensure there are enough entries to sample
    if n > len(data):
        raise ValueError(f"Requested {n} entries, but the file only contains {len(data)} entries.")

    # Randomly sample N entries without replacement
    selected_entries = random.sample(data, n)

    # Write the sampled entries to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(selected_entries, outfile, indent=4, ensure_ascii=False)

    print(f"Selected {n} random entries written to {output_file}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Select N random entries from a JSON file.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input JSON file")
    parser.add_argument("-o", "--output", required=True, help="Path to the output JSON file")
    parser.add_argument("-n", "--number", required=True, type=int, help="Number of random entries to select")

    args = parser.parse_args()

    try:
        select_random_entries(args.input, args.output, args.number)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
