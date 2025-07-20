"""Utilities for loading and cleaning dictionary data."""
import json
from pathlib import Path
from typing import List, Dict


def load_dictionary(path: str) -> List[Dict]:
    """Load a dictionary file filtering out entries without translation."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dictionary not found: {path}")
    with p.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return [entry for entry in data if entry.get('translation', '').strip()]


def save_dictionary(entries: List[Dict], dialect: str, output_dir: str = "Dictionary") -> Path:
    """Save cleaned dictionary entries to a standard path."""
    out_path = Path(output_dir) / f"{dialect}Dictionary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    return out_path
