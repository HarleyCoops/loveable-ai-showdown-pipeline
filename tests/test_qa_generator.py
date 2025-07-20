"""
Unit tests for the BilingualQAGenerator class.
These examples illustrate best practices like stubbing external dependencies
and verifying that helper methods behave as expected.
"""
import json
import sys
import tempfile
from pathlib import Path
import types
import unittest

# Provide a minimal stub for the openai module so the import succeeds
openai_stub = types.SimpleNamespace(OpenAI=object)
sys.modules.setdefault('openai', openai_stub)

# Make the Scripts directory importable
sys.path.append(str(Path(__file__).resolve().parents[1] / 'Scripts'))

from openAI_bilingual_qa_generator import BilingualQAGenerator


class TestBilingualQAGenerator(unittest.TestCase):
    """Tests for dictionary loading and prompt creation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create a small dictionary file with some missing translations
        self.dict_file = self.temp_path / 'dict.json'
        entries = [
            {'word': 'dog', 'translation': 'hound'},
            {'word': 'cat', 'translation': ''},
            {'word': 'bird'}  # missing translation key
        ]
        self.dict_file.write_text(json.dumps(entries), encoding='utf-8')

        self.output_file = self.temp_path / 'out.jsonl'

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_dictionary_filters_missing_translations(self):
        generator = BilingualQAGenerator('TestDialect', str(self.dict_file), str(self.output_file))
        # Only one entry has a non-empty translation
        self.assertEqual(len(generator.dictionary_entries), 1)
        self.assertEqual(generator.dictionary_entries[0]['word'], 'dog')

    def test_create_context_prompt_contains_entries(self):
        generator = BilingualQAGenerator('TestDialect', str(self.dict_file), str(self.output_file))
        prompt = generator.create_context_prompt(generator.dictionary_entries)
        # Each dictionary entry should be embedded in the prompt text
        for entry in generator.dictionary_entries:
            self.assertIn(json.dumps(entry, ensure_ascii=False), prompt)
        # The dialect name is also referenced in the instructions
        self.assertIn('TestDialect', prompt)


if __name__ == '__main__':
    unittest.main()
