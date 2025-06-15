
#!/usr/bin/env python3
"""
Convert QA JSONL to OpenAI chat-format JSONL for fine-tuning.
Takes our dialect-specific QA pairs and converts them to the format required by OpenAI.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict


def convert_qa_to_chat_format(input_file: str, output_file: str, dialect: str) -> None:
    """
    Convert a QA JSONL file (question/answer) into the OpenAI chat-style format.

    Args:
        input_file: Path to input JSONL file with question/answer pairs
        output_file: Path to output JSONL file with chat messages
        dialect: Name of the dialect (e.g., "Thlinkit_Skutkwan")
    """
    system_prompt = (
        f"You are an assistant expert in the {dialect} dialect. "
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
                        # Already in chat format, keep as is
                        converted.append(entry)
                    else:
                        # Convert QA pair to chat format
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

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save in OpenAI's preferred format (compact JSON)
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in converted:
            outfile.write(json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n')
            
    print(f"Done. Converted {len(converted)} entries.")


def prepare_fine_tuning_data(input_file: str, output_base: str, train_ratio: float = 0.8) -> None:
    """
    Split the converted data into training and validation sets.

    Args:
        input_file: Path to the converted JSONL file
        output_base: Base name for output files (will create _train.jsonl and _valid.jsonl)
        train_ratio: Ratio of data to use for training (default: 0.8)
    """
    # Read all entries
    entries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    # Shuffle and split
    import random
    random.shuffle(entries)
    split_idx = int(len(entries) * train_ratio)
    train_data = entries[:split_idx]
    valid_data = entries[split_idx:]

    # Save training set
    train_file = f"{output_base}_train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n')

    # Save validation set
    valid_file = f"{output_base}_valid.jsonl"
    with open(valid_file, 'w', encoding='utf-8') as f:
        for entry in valid_data:
            f.write(json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n')

    print(f"Split data into:")
    print(f"  Training set: {len(train_data)} entries → {train_file}")
    print(f"  Validation set: {len(valid_data)} entries → {valid_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert QA JSONL to OpenAI fine-tuning format'
    )
    parser.add_argument('--dialect', required=True, help='Name of the dialect (e.g., Thlinkit_Skutkwan)')
    parser.add_argument('--input', required=True, help='Input JSONL file with question/answer pairs')
    parser.add_argument('--output', required=True, help='Base name for output files')
    args = parser.parse_args()

    # First convert to chat format
    temp_file = f"{args.output}_converted.jsonl"
    convert_qa_to_chat_format(args.input, temp_file, args.dialect)

    # Then prepare training/validation split
    prepare_fine_tuning_data(temp_file, args.output)

    # Clean up temporary file
    os.remove(temp_file)
