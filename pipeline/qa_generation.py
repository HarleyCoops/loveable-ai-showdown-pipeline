"""QA generation utilities supporting different model providers."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional

from .data_ingestion import load_dictionary


class BaseQAGenerator:
    """Abstract base class for generating QA pairs."""

    def __init__(self, dialect_name: str, entries: List[Dict], target_count: int = 500):
        self.dialect_name = dialect_name
        self.entries = entries
        self.target_count = target_count

    def create_prompt(self, batch: List[Dict]) -> str:
        context = (
            f"You are a linguist generating educational question-answer pairs for {self.dialect_name}. "
            "Use only the provided dictionary entries. Return a JSON array of objects with 'question' and 'answer'."
        )
        for entry in batch:
            context += "\n" + json.dumps(entry, ensure_ascii=False)
        context += "\n\nGenerate the JSON array now."
        return context

    def generate(self, batch_size: int = 10):
        raise NotImplementedError


class OpenAIQAGenerator(BaseQAGenerator):
    """Generator that calls the OpenAI API."""

    def __init__(self, dialect_name: str, entries: List[Dict], target_count: int = 500):
        super().__init__(dialect_name, entries, target_count)
        from openai import OpenAI  # imported lazily for tests
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)

    def _call_model(self, prompt: str) -> Optional[str]:
        try:
            response = self.client.responses.create(
                model="o3-pro",
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                text={"format": {"type": "text"}},
                reasoning={"effort": "medium", "summary": "auto"},
                tools=[],
                store=True,
            )
            if (
                hasattr(response, "output")
                and len(response.output) > 1
                and hasattr(response.output[1], "content")
                and len(response.output[1].content) > 0
                and hasattr(response.output[1].content[0], "text")
            ):
                return response.output[1].content[0].text
            return None
        except Exception as exc:  # pragma: no cover - network call
            print(f"OpenAI API call failed: {exc}")
            return None

    def generate(self, batch_size: int = 10, output_file: Optional[str] = None):
        if not self.entries:
            print("No dictionary entries to process.")
            return
        path = Path(output_file) if output_file else None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")

        random.shuffle(self.entries)
        total = 0
        batch_index = 0
        while total < self.target_count:
            start = (batch_index * batch_size) % len(self.entries)
            batch = self.entries[start:start + batch_size]
            if not batch:
                random.shuffle(self.entries)
                continue
            batch_index += 1
            prompt = self.create_prompt(batch)
            text = self._call_model(prompt)
            if text:
                try:
                    qa_pairs = json.loads(text)
                    if isinstance(qa_pairs, list):
                        if path:
                            with path.open("a", encoding="utf-8") as f:
                                for qa in qa_pairs:
                                    f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                        total += len(qa_pairs)
                        print(f"Generated {len(qa_pairs)} QA pairs (total {total})")
                    else:
                        print("Warning: response was not a list")
                except json.JSONDecodeError:
                    print("Failed to decode JSON; skipping batch")
            if total >= self.target_count:
                break


# Skeleton for a Gemma-based generator
class GemmaQAGenerator(BaseQAGenerator):
    """Use a local Gemma model via transformers to generate QA pairs."""

    def __init__(self, dialect_name: str, entries: List[Dict], model_name: str = "google/gemma-2b", target_count: int = 500, device: str = "cpu"):
        super().__init__(dialect_name, entries, target_count)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.device = device

    def _generate_text(self, prompt: str) -> str:
        from transformers import TextStreamer
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.8, top_p=0.9)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate(self, batch_size: int = 10, output_file: Optional[str] = None):
        if not self.entries:
            print("No dictionary entries to process.")
            return
        path = Path(output_file) if output_file else None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")

        random.shuffle(self.entries)
        total = 0
        batch_index = 0
        while total < self.target_count:
            start = (batch_index * batch_size) % len(self.entries)
            batch = self.entries[start:start + batch_size]
            if not batch:
                random.shuffle(self.entries)
                continue
            batch_index += 1
            prompt = self.create_prompt(batch)
            text = self._generate_text(prompt)
            try:
                qa_pairs = json.loads(text[text.find("[") : text.rfind("]") + 1])
                if isinstance(qa_pairs, list):
                    if path:
                        with path.open("a", encoding="utf-8") as f:
                            for qa in qa_pairs:
                                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                    total += len(qa_pairs)
                    print(f"Generated {len(qa_pairs)} QA pairs (total {total})")
            except Exception:
                print("Failed to parse Gemma output; skipping batch")
            if total >= self.target_count:
                break


# Simple wrapper matching the original interface
class BilingualQAGenerator(OpenAIQAGenerator):
    """Compat class that loads a dictionary from disk and writes to a file."""

    def __init__(self, dialect_name: str, input_path: str, output_path: str):
        entries = load_dictionary(input_path)
        super().__init__(dialect_name, entries)
        self.output_path = output_path
        self.dictionary_entries = entries
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("")

    def generate(self, batch_size: int = 10):
        super().generate(batch_size=batch_size, output_file=self.output_path)

    # Backwards compatibility with previous API
    def create_context_prompt(self, entries: List[Dict]) -> str:  # pragma: no cover
        return self.create_prompt(entries)


