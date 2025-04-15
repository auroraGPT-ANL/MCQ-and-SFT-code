#!/usr/bin/env python

import json
import argparse

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Extract question and answer fields from a JSON file.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input JSON file")
    parser.add_argument("-o", "--output", required=True, help="Path to the output JSON file")
    args = parser.parse_args()

    input_json = args.input
    output_json = args.output

    # Read the JSON data from the input file
    with open(input_json, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    # Extract only the question and answer keys
    results = []
    for record in data:
        # Assuming each record has keys: "question", "answer", and "text"
        new_record = {
            "question": record.get("question"),
            "answer": record.get("answer")
        }
        results.append(new_record)

    # Write the extracted data to the output JSON file
    with open(output_json, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4, ensure_ascii=False)

    print(f"Extracted data written to {output_json}")

if __name__ == "__main__":
    main()
