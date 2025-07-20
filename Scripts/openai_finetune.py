#!/usr/bin/env python3
"""Wrapper for OpenAI fine tuning module."""
import argparse
from pipeline.openai_finetune import OpenAIFineTuner


def main() -> None:
    parser = argparse.ArgumentParser(description='Run OpenAI fine-tuning')
    parser.add_argument('--dialect', required=False)
    args = parser.parse_args()

    if args.dialect:
        OpenAIFineTuner(args.dialect).run()
    else:
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        for dict_file in (root / 'Dictionary').glob('*Dictionary.json'):
            dialect = dict_file.stem.replace('Dictionary', '')
            OpenAIFineTuner(dialect).run()


if __name__ == '__main__':
    main()

