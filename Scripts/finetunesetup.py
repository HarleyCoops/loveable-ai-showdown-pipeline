
import json
import glob
import random
from pathlib import Path

def prepare_fine_tuning_data(input_file: str, output_file: str, dialect: str):
    """
    Prepare fine-tuning data by splitting into train/validation sets.
    
    Args:
        input_file: Path to the input JSONL file with chat messages
        output_file: Base name for output files (will create _train.jsonl and _valid.jsonl)
        dialect: Name of the dialect being processed
    """
    data = []
    print(f"Processing {dialect} data from {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if 'messages' in entry:
                        data.append(entry)
                    else:
                        print(f"Warning: Skipping entry without 'messages' field")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line")
                except Exception as e:
                    print(f"Warning: Error processing line: {e}")
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return

    if not data:
        print("Error: No valid data found in input file")
        return

    # Shuffle the data
    random.shuffle(data)
    
    # Split into train/validation sets (80/20 split)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    valid_data = data[split_idx:]

    # Create output directory if it doesn't exist
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training set
    train_file = f"{output_file}_train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Training file created: {train_file} ({len(train_data)} examples)")

    # Save validation set
    valid_file = f"{output_file}_valid.jsonl"
    with open(valid_file, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Validation file created: {valid_file} ({len(valid_data)} examples)")

if __name__ == '__main__':
    # Get the project root by going up one directory from the script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Define paths
    input_file = project_root / 'Output' / 'finetune_qa_Thlinkit_Skutkwan.jsonl'
    output_base = project_root / 'Output' / 'finetune_Thlinkit_Skutkwan'
    
    # Process the data
    prepare_fine_tuning_data(
        input_file=str(input_file),
        output_file=str(output_base),
        dialect="Thlinkit_Skutkwan"
    )
