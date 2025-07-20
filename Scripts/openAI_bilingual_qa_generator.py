#!/usr/bin/env python3
"""Wrapper script calling the package-based QA generator."""
import argparse
from pathlib import Path

from pipeline.data_ingestion import load_dictionary
from pipeline.qa_generation import OpenAIQAGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic QA pairs using OpenAI")
    parser.add_argument('--dialect-name', required=False, help='Dialect name')
    parser.add_argument('--input', required=False, help='Path to dictionary JSON')
    parser.add_argument('--output', required=False, help='Output JSONL file')
    parser.add_argument('--target-count', type=int, default=500)
    args = parser.parse_args()

    if args.dialect_name and args.input and args.output:
        entries = load_dictionary(args.input)
        gen = OpenAIQAGenerator(args.dialect_name, entries, target_count=args.target_count)
        gen.generate(output_file=args.output)
    else:
        # fallback: process all dictionaries in Dictionary/
        root = Path(__file__).resolve().parents[1]
        for dict_file in (root / 'Dictionary').glob('*Dictionary.json'):
            dialect = dict_file.stem.replace('Dictionary', '')
            output = dict_file.parent / f'synthetic_qa_{dialect}_openai.jsonl'
            entries = load_dictionary(str(dict_file))
            gen = OpenAIQAGenerator(dialect, entries, target_count=args.target_count)
            gen.generate(output_file=str(output))


if __name__ == '__main__':
    main()

