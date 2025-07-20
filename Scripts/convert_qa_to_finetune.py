#!/usr/bin/env python3
"""Wrapper for dataset conversion utilities."""
import argparse

from pipeline.dataset_conversion import convert_qa_to_chat_format, prepare_fine_tuning_data


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert QA data to fine-tune format')
    parser.add_argument('--dialect', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    args = parser.parse_args()

    temp = args.output + '_converted.jsonl'
    convert_qa_to_chat_format(args.input, temp, args.dialect)
    prepare_fine_tuning_data(temp, args.output, train_ratio=args.train_ratio)


if __name__ == '__main__':
    main()

