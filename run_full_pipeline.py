
#!/usr/bin/env python3
"""
run_full_pipeline.py - Orchestrate the full data pipeline for LoveableAIShowdown.

This script automates the end-to-end workflow for each dialect:
1. Generate synthetic QA pairs from each Dictionary JSON.
2. Convert synthetic QA to OpenAI chat-format and split into train/valid sets.
3. Fine-tune an OpenAI model on each dialect's train/valid data.

Usage:
    python run_full_pipeline.py

Requirements:
    - Python 3.8+
    - All dependencies installed (`pip install -r requirements.txt -r requirements-deploy.txt`)
    - .env configured with OPENAI_API_KEY, WANDB_API_KEY, HF_TOKEN, etc.
"""
import subprocess
import sys
from pathlib import Path

def main():
    root = Path(__file__).parent.resolve()
    dict_dir = root / "Dictionary"
    output_dir = root / "Output"

    # 1. Synthetic QA generation for all dialects
    print("\n=== Step 1/3: Synthetic QA generation ===")
    subprocess.run([
        sys.executable,
        "Scripts/OpenAI_bilingual_qa_generator.py"
    ], check=True)

    # 2. Convert QA to chat-format and split
    print("\n=== Step 2/3: Convert to fine-tuning format ===")
    for dict_file in sorted(dict_dir.glob("*Dictionary.json")):
        dialect = dict_file.stem.replace("Dictionary", "")
        in_file = output_dir / f"synthetic_qa_{dialect}_openai.jsonl"
        out_base = output_dir / f"finetune_qa_{dialect}"
        print(f"\n-- Processing dialect: {dialect} --")
        subprocess.run([
            sys.executable,
            "Scripts/convert_qa_to_finetune.py",
            "--dialect", dialect,
            "--input", str(in_file),
            "--output", str(out_base)
        ], check=True)

    # 3. Fine-tuning for all dialects
    print("\n=== Step 3/3: Fine-tune models ===")
    subprocess.run([
        sys.executable,
        "Scripts/openai_finetune.py"
    ], check=True)

    print("\n✅ Full pipeline completed successfully.")

if __name__ == "__main__":
    main()
