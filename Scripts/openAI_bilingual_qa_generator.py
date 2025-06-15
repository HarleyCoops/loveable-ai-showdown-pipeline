
import json
import os
import random
from openai import OpenAI
from typing import List, Dict, Any, Optional
from pathlib import Path

def call_llm_api(prompt: str) -> Optional[str]:
    """
    Calls the OpenAI O3 Pro API and returns the model's raw JSON string response.

    Args:
        prompt: Fully formatted prompt string.

    Returns:
        Raw text returned by the model (expected JSON) or None on failure.
    """
    try:
        # 1. API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(
                "Error: OPENAI_API_KEY environment variable not set.\n"
                "Create one at https://platform.openai.com/api-keys "
                "and export OPENAI_API_KEY=your_key"
            )
            return None
        client = OpenAI(api_key=api_key)

        # 2. Call OpenAI O3 Pro API
        print("--- CALLING OPENAI O3 PRO API ---")
        response = client.responses.create(
            model="o3-pro",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                        }
                    ]
                }
            ],
            text={
                "format": {
                    "type": "text"
                }
            },
            reasoning={
                "effort": "medium",
                "summary": "auto"
            },
            tools=[],
            store=True
        )

        # 3. Extract the assistant's output
        if (hasattr(response, 'output') and 
            len(response.output) > 1 and 
            hasattr(response.output[1], 'content') and 
            len(response.output[1].content) > 0 and 
            hasattr(response.output[1].content[0], 'text')):
            return response.output[1].content[0].text
        else:
            print("Could not find text in response structure")
            return None

    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        return None


class BilingualQAGenerator:
    """
    Generates bilingual Question-Answer pairs from a source dictionary file
    by leveraging a Large Language Model.
    """
    def __init__(self, dialect_name: str, input_path: str, output_path: str):
        self.dialect_name = dialect_name
        self.input_path = input_path
        self.output_path = output_path
        self.dictionary_entries = self._load_dictionary()
        self.target_qa_count = 500  # Target number of QA pairs to generate
        # Create output directory and initialize the JSONL file
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        # Clear the file if it exists
        open(self.output_path, 'w').close()

    def _load_dictionary(self) -> List[Dict]:
        """Loads and filters the dictionary from the source JSON file."""
        print(f"Loading dictionary from: {self.input_path}")
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Filter out entries with no translation
            filtered_data = [
                entry for entry in data if entry.get('translation', '').strip()
            ]
            print(f"Loaded {len(data)} entries, filtered to {len(filtered_data)} with translations.")
            return filtered_data
        except FileNotFoundError:
            print(f"Error: Dictionary file not found at {self.input_path}")
            return []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.input_path}")
            return []

    def create_context_prompt(self, entries: List[Dict]) -> str:
        """
        Creates a detailed, instruction-rich prompt for the LLM.
        This is the core of the generation logic.
        """
        context = f"""You are a linguist and an expert in generating educational material for the {self.dialect_name} dialect of Tlingit, based on a historical wordlist.
Your task is to create a diverse set of natural question-answer pairs using ONLY the information from the provided dictionary entries. The output MUST be a valid JSON array of objects, where each object has a "question" key and an "answer" key.

**CRITICAL GUIDELINES:**
1.  **Source Fidelity:** Adhere strictly to the provided data. DO NOT invent new words, alternate spellings, or information not present. Use the exact phonetic spelling from the 'translation' field.
2.  **Output Format:** Your entire response must be a single JSON array `[...]` containing `{{ "question": "...", "answer": "..." }}` objects. Do not include any text or explanations outside of this JSON structure.
3.  **Question Diversity:** For the given batch of words, generate multiple QA pairs covering different angles:
    - **Direct Translation:** (e.g., "What is the English for '...'?", "How do you say '...' in {self.dialect_name}?")
    - **Contextual Usage:** Look for hints like "(said by son)" or "(black)" and create scenario questions. (e.g., "If a son is talking about his father, what word does he use?")
    - **Semantic Relationships:** If the batch contains related words (e.g., 'Man'/'Woman', 'Day'/'Night', 'Alive'/'Dead'), ask comparative or relational questions.
    - **Categorization:** Ask questions that require identifying a word's category. (e.g., "Which of these words is a type of animal?")
    - **Component Analysis (if obvious):** If a word seems to be a compound of another word in the list (e.g., 'shā-wut-uh-ghutti' contains 'shā-wut'), create a question about it.

**DICTIONARY ENTRIES (Ground Truth):**
"""
        
        for entry in entries:
            context += f"\n{json.dumps(entry, ensure_ascii=False)}"
        
        context += "\n\nNow, generate the JSON array of question-answer pairs based on these entries and the guidelines."
        return context

    def _save_batch(self, qa_pairs: List[Dict]):
        """Saves a batch of QA pairs to the output file in JSONL format."""
        with open(self.output_path, 'a', encoding='utf-8') as f:
            for qa_pair in qa_pairs:
                f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')

    def generate(self, batch_size: int = 10):
        """
        Processes the dictionary in batches and generates QA pairs until reaching the target count.

        Args:
            batch_size: The number of dictionary entries to include in each prompt.
        """
        if not self.dictionary_entries:
            print("No dictionary entries to process.")
            return

        total_qa_pairs = 0
        batch_count = 0
        
        # Shuffle entries to get diverse batches each run
        random.shuffle(self.dictionary_entries)
        
        # Process batches until we reach the target count or run out of entries
        while total_qa_pairs < self.target_qa_count:
            # Get next batch of entries
            start_idx = (batch_count * batch_size) % len(self.dictionary_entries)
            batch = self.dictionary_entries[start_idx:start_idx + batch_size]
            
            if not batch:  # If we've gone through all entries, reshuffle and continue
                random.shuffle(self.dictionary_entries)
                continue
                
            batch_count += 1
            print(f"\n--- Processing Batch {batch_count} ---")
            
            # 1. Create the prompt for the current batch
            prompt = self.create_context_prompt(batch)
            
            # 2. Call the LLM API
            response_text = call_llm_api(prompt)

            # 3. Process the response
            if response_text:
                try:
                    # The LLM should return a clean JSON string
                    qa_pairs = json.loads(response_text)
                    if isinstance(qa_pairs, list):
                        # Save this batch immediately
                        self._save_batch(qa_pairs)
                        total_qa_pairs += len(qa_pairs)
                        print(f"Successfully generated {len(qa_pairs)} QA pairs for this batch.")
                        print(f"Total QA pairs so far: {total_qa_pairs}")
                    else:
                        print("Warning: LLM response was valid JSON but not a list.")
                except json.JSONDecodeError:
                    print("Error: Failed to decode JSON from LLM response. Skipping batch.")
                    print("--- LLM Raw Response ---")
                    print(response_text)
                    print("------------------------")
            else:
                print("Error: No response from LLM API. Skipping batch.")

            # Stop if we've reached the target count
            if total_qa_pairs >= self.target_qa_count:
                break

        print(f"\nCompleted generating {total_qa_pairs} QA pairs.")
        print(f"Output saved to: {self.output_path}")


def process_all_dialects():
    """Process all dialect dictionary files in the Dictionary directory."""
    # Get the project root by going up one directory from the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dictionary_dir = os.path.join(project_root, 'Dictionary')
    
    # Find all dictionary files (pure JSON files)
    dictionary_files = list(Path(dictionary_dir).glob('*Dictionary.json'))
    
    for dict_file in dictionary_files:
        dialect_name = dict_file.stem.replace('Dictionary.json', '')
        output_file = os.path.join(dictionary_dir, f'synthetic_qa_{dialect_name}_openai.jsonl')
        
        print(f"\nProcessing dialect: {dialect_name}")
        generator = BilingualQAGenerator(
            dialect_name=dialect_name,
            input_path=str(dict_file),
            output_path=output_file
        )
        generator.generate(batch_size=10)


if __name__ == '__main__':
    # When run as a script, process all dialect dictionaries in the Dictionary folder
    process_all_dialects()
