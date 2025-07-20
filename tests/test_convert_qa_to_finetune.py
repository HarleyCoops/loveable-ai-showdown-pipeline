"""
Example unit tests for the QA conversion utilities.
Each test includes detailed comments explaining what is being tested
and demonstrates best practices such as using temporary directories
and deterministic randomness.
"""
import json
import os
import random
import sys
import tempfile
from pathlib import Path
import unittest

# Allow importing modules from the Scripts directory
sys.path.append(str(Path(__file__).resolve().parents[1] / 'Scripts'))

from convert_qa_to_finetune import convert_qa_to_chat_format, prepare_fine_tuning_data


class TestConvertQA(unittest.TestCase):
    """Unit tests for conversion utilities with instructional comments."""

    def setUp(self):
        # Create a temporary directory for our test files. Using TemporaryDirectory
        # ensures files are cleaned up even if a test fails.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Sample QA data we'll convert. Each line is a JSON object.
        self.sample_input = self.temp_path / 'sample_qa.jsonl'
        with open(self.sample_input, 'w', encoding='utf-8') as f:
            # A simple question/answer pair
            f.write(json.dumps({'question': 'Hello?', 'answer': 'Hi'}) + '\n')
            # Already-formatted chat data should pass through unchanged
            f.write(json.dumps({
                'messages': [
                    {'role': 'user', 'content': 'Ping?'},
                    {'role': 'assistant', 'content': 'Pong!'}
                ]
            }) + '\n')

    def tearDown(self):
        # Always clean up to avoid leaving temp files on disk.
        self.temp_dir.cleanup()

    def test_convert_qa_to_chat_format(self):
        """Ensure QA pairs are converted to OpenAI chat format."""
        output_file = self.temp_path / 'converted.jsonl'
        convert_qa_to_chat_format(str(self.sample_input), str(output_file), 'TestDialect')

        lines = [json.loads(l) for l in output_file.read_text().splitlines() if l]
        self.assertEqual(len(lines), 2)

        first = lines[0]
        # Expect three messages: system, user, assistant
        self.assertIn('messages', first)
        self.assertEqual(len(first['messages']), 3)
        self.assertEqual(first['messages'][1]['content'], 'Hello?')
        self.assertEqual(first['messages'][2]['content'], 'Hi')

        second = lines[1]
        # The second entry already contained messages; it should remain unchanged
        self.assertEqual(second['messages'][0]['content'], 'Ping?')
        self.assertEqual(second['messages'][1]['content'], 'Pong!')

    def test_prepare_fine_tuning_data(self):
        """Verify data is split into train/valid files."""
        converted = self.temp_path / 'converted.jsonl'
        convert_qa_to_chat_format(str(self.sample_input), str(converted), 'TestDialect')

        # Seed the RNG so shuffling is deterministic during tests
        random.seed(0)
        prepare_fine_tuning_data(str(converted), str(self.temp_path / 'out'), train_ratio=0.5)

        train_file = self.temp_path / 'out_train.jsonl'
        valid_file = self.temp_path / 'out_valid.jsonl'

        self.assertTrue(train_file.exists())
        self.assertTrue(valid_file.exists())

        train_lines = train_file.read_text().splitlines()
        valid_lines = valid_file.read_text().splitlines()
        # With two total entries and a 0.5 split, each file should have one line
        self.assertEqual(len(train_lines), 1)
        self.assertEqual(len(valid_lines), 1)


if __name__ == '__main__':
    # Running tests via `python -m unittest` will execute this file as well.
    unittest.main()
