#!/usr/bin/env python

import json
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Extract only the 'text' field from a JSONL file."
    )
    parser.add_argument("-i", "--input",  required=True, help="Path to input .jsonl")
    parser.add_argument("-o", "--output", required=True, help="Path to output .jsonl")
    args = parser.parse_args()

    results = []
    with open(args.input, 'r', encoding='utf-8') as infile:
        for line in infile:
            if not line.strip():
                continue
            record = json.loads(line)
            results.append({"text": record.get("text")})

    # If you want a single JSON array as output:
    with open(args.output, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4, ensure_ascii=False)

    # Or, if you’d rather keep it JSONL (one {"text":…} per line), do:
    # with open(args.output, 'w', encoding='utf-8') as outfile:
    #     for rec in results:
    #         outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Extracted {len(results)} records to {args.output}")

if __name__ == "__main__":
    main()

