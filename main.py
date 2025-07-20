"""Command line interface for the Loveable AI pipeline."""
import argparse
from pathlib import Path

from pipeline.data_ingestion import load_dictionary, save_dictionary
from pipeline.qa_generation import OpenAIQAGenerator, GemmaQAGenerator
from pipeline.dataset_conversion import convert_qa_to_chat_format, prepare_fine_tuning_data
from pipeline.openai_finetune import OpenAIFineTuner
from pipeline.lora_training import LoraConfig, train_lora


def cmd_ingest(args: argparse.Namespace) -> None:
    entries = load_dictionary(args.input)
    save_dictionary(entries, args.dialect_name, args.output_dir)
    print(f"Saved cleaned dictionary for {args.dialect_name}")


def cmd_generate(args: argparse.Namespace) -> None:
    entries = load_dictionary(args.input)
    if args.provider == 'gemma':
        generator = GemmaQAGenerator(args.dialect_name, entries, target_count=args.target_count)
    else:
        generator = OpenAIQAGenerator(args.dialect_name, entries, target_count=args.target_count)
    generator.generate(batch_size=10, output_file=args.output)


def cmd_convert(args: argparse.Namespace) -> None:
    convert_qa_to_chat_format(args.input, args.output + '_converted.jsonl', args.dialect)
    prepare_fine_tuning_data(args.output + '_converted.jsonl', args.output, train_ratio=args.train_ratio)
    Path(args.output + '_converted.jsonl').unlink()


def cmd_finetune_openai(args: argparse.Namespace) -> None:
    tuner = OpenAIFineTuner(args.dialect)
    tuner.run()


def cmd_finetune_gemma(args: argparse.Namespace) -> None:
    config = LoraConfig(model_name=args.model)
    train_lora(config, args.dataset, args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Loveable AI pipeline")
    sub = parser.add_subparsers(dest='command')

    p_ingest = sub.add_parser('ingest')
    p_ingest.add_argument('--dialect-name', required=True)
    p_ingest.add_argument('--input', required=True)
    p_ingest.add_argument('--output-dir', default='Dictionary')
    p_ingest.set_defaults(func=cmd_ingest)

    p_gen = sub.add_parser('generate_qa')
    p_gen.add_argument('--dialect-name', required=True)
    p_gen.add_argument('--input', required=True)
    p_gen.add_argument('--output', required=True)
    p_gen.add_argument('--target-count', type=int, default=500)
    p_gen.add_argument('--provider', choices=['openai', 'gemma'], default='openai')
    p_gen.set_defaults(func=cmd_generate)

    p_conv = sub.add_parser('convert')
    p_conv.add_argument('--dialect', required=True)
    p_conv.add_argument('--input', required=True)
    p_conv.add_argument('--output', required=True)
    p_conv.add_argument('--train-ratio', type=float, default=0.8)
    p_conv.set_defaults(func=cmd_convert)

    p_fto = sub.add_parser('finetune_openai')
    p_fto.add_argument('--dialect', required=True)
    p_fto.set_defaults(func=cmd_finetune_openai)

    p_ftg = sub.add_parser('finetune_gemma')
    p_ftg.add_argument('--model', required=True)
    p_ftg.add_argument('--dataset', required=True)
    p_ftg.add_argument('--output-dir', required=True)
    p_ftg.set_defaults(func=cmd_finetune_gemma)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        return
    args.func(args)


if __name__ == '__main__':
    main()

