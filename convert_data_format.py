
#!/usr/bin/env python3
"""
Convert QA JSONL to OpenAI chat-format JSONL for fine-tuning.
"""
import argparse
import json
import os
from pathlib import Path


def convert_file_format(input_file: str, output_file: str, dialect: str) -> None:
    """
    Convert a QA JSONL file (question/answer) into the OpenAI chat-style format.

    Each input line must be JSON with 'question' and 'answer' fields. The output
    will be a JSONL file where each entry has a 'messages' array suitable for fine-tuning.
    """
    system_prompt = (
        f"You are an assistant expert in the {dialect} dialect of Tlingit. "
        f"Provide concise answers or explanations in {dialect}."
    )
    print(f"Converting '{input_file}' → '{output_file}' (dialect={dialect})...")
    converted = []
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            for idx, line in enumerate(infile, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if 'messages' in entry:
                        converted.append(entry)
                    else:
                        q = entry['question']
                        a = entry['answer']
                        converted.append({
                            'messages': [
                                {'role': 'system',    'content': system_prompt},
                                {'role': 'user',      'content': q},
                                {'role': 'assistant', 'content': a},
                            ]
                        })
                except json.JSONDecodeError:
                    print(f"  [warning] invalid JSON on line {idx}")
                except KeyError as e:
                    print(f"  [warning] missing key {e} on line {idx}")
    except FileNotFoundError:
        print(f"Error: input file not found: {input_file}")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in converted:
            outfile.write(json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(f"Done. Converted {len(converted)} entries.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert QA JSONL to OpenAI messages format'
    )
    parser.add_argument('--dialect', required=True, help='Name of the dialect (e.g. Thlinkit_Skutkwan)')
    parser.add_argument('--input',   required=True, help='Input JSONL file with question/answer fields')
    parser.add_argument('--output',  required=True, help='Output JSONL file with chat messages')
    args = parser.parse_args()
    convert_file_format(args.input, args.output, args.dialect)
