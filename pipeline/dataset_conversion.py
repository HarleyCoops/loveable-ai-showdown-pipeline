"""Utilities for converting QA data into fine-tuning datasets."""
import json
import os
from typing import List
from pathlib import Path


def convert_qa_to_chat_format(input_file: str, output_file: str, dialect: str) -> None:
    system_prompt = f"You are an assistant expert in the {dialect} dialect. Provide concise answers in {dialect}."
    converted: List[dict] = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for idx, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if 'messages' in entry:
                converted.append(entry)
            else:
                converted.append({
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': entry['question']},
                        {'role': 'assistant', 'content': entry['answer']},
                    ]
                })
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as out:
        for entry in converted:
            out.write(json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n')


def prepare_fine_tuning_data(input_file: str, output_base: str, train_ratio: float = 0.8) -> None:
    entries = [json.loads(l) for l in open(input_file, 'r', encoding='utf-8').read().splitlines() if l]
    import random
    random.shuffle(entries)
    split = int(len(entries) * train_ratio)
    train, valid = entries[:split], entries[split:]
    train_file = f"{output_base}_train.jsonl"
    valid_file = f"{output_base}_valid.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for e in train:
            f.write(json.dumps(e, ensure_ascii=False, separators=(',', ':')) + '\n')
    with open(valid_file, 'w', encoding='utf-8') as f:
        for e in valid:
            f.write(json.dumps(e, ensure_ascii=False, separators=(',', ':')) + '\n')

