# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hybrid React/TypeScript frontend with Python backend project for training low-resource language models. It processes dictionaries of First Nations languages (Haida, Tlingit, Tshimshian dialects) through a pipeline that generates synthetic data, fine-tunes OpenAI models, and deploys interactive Gradio apps on Hugging Face Spaces.

## Essential Commands

### Frontend Development (React/Vite)
```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run linting
npm run lint

# Preview production build
npm run preview
```

### Python Pipeline Commands
```bash
# Install Python dependencies
pip install -r requirements.txt -r requirements-deploy.txt

# Run complete pipeline (all dialects)
python run_full_pipeline.py

# Individual pipeline steps:
# 1. Generate synthetic QA pairs
python Scripts/openAI_bilingual_qa_generator.py

# 2. Convert to fine-tuning format (replace <Dialect> with actual name)
python Scripts/convert_qa_to_finetune.py \
  --dialect <Dialect> \
  --input Dictionary/synthetic_qa_<Dialect>_openai.jsonl \
  --output Output/finetune_qa_<Dialect>

# 3. Fine-tune models
python Scripts/openai_finetune.py

# Deploy to Hugging Face Spaces
python HFSpacesDeployment.py --dialect-name "Thlinkit_Skutkwan" --verbose
```

## High-Level Architecture

### Project Structure
The project has two main components:

1. **React Frontend** (src/): A Vite-based React application with TypeScript and Tailwind CSS
   - Uses shadcn/ui components for consistent UI
   - Main entry: src/main.tsx → src/App.tsx
   - Pages in src/pages/, reusable components in src/components/ui/

2. **Python Backend Pipeline**: Processes language dictionaries through multiple stages
   - Dictionary/ - Source JSON files with native words and translations
   - Scripts/ - Core pipeline scripts
   - Output/ - Generated training data and fine-tuning outputs
   - HFSpacesApp/ - Templates for Hugging Face deployment

### Data Flow Pipeline

```
Dictionary JSON files (native words + translations)
    ↓ (openAI_bilingual_qa_generator.py)
Synthetic QA pairs generation using OpenAI O3 Pro
    ↓ (convert_qa_to_finetune.py)
Chat-formatted training data with 80/20 train/validation split
    ↓ (openai_finetune.py)
Fine-tuned GPT-4.1 models with dialect-specific knowledge
    ↓ (HFSpacesDeployment.py)
Interactive Gradio chat apps deployed on Hugging Face Spaces
```

### Key Pipeline Components

1. **BilingualQAGenerator** (openAI_bilingual_qa_generator.py):
   - Processes dictionary entries in batches of 10
   - Generates 500 diverse QA pairs per dialect
   - Uses OpenAI O3 Pro with "medium" reasoning effort

2. **Data Conversion** (convert_qa_to_finetune.py):
   - Converts QA pairs to OpenAI chat format
   - Adds dialect-specific system prompts
   - Creates 80/20 train/validation split

3. **OpenAIFineTuner** (openai_finetune.py):
   - Fine-tunes GPT-4.1 models (3 epochs, batch size 4)
   - Integrates with Weights & Biases for tracking
   - Updates .env with fine-tuned model IDs

4. **HFSpacesDeployer** (HFSpacesDeployment.py):
   - Creates Gradio chat interfaces
   - Deploys to Hugging Face Spaces via Git
   - Supports both personal and organization deployments

### Environment Configuration
Required .env variables:
- OPENAI_API_KEY - For API access and fine-tuning
- WANDB_API_KEY - For experiment tracking
- HF_TOKEN - For Hugging Face deployment
- Fine-tuned model IDs (auto-populated by pipeline)

### Supported Dialects
- Thlinkit_Skutkwan
- Haida_Kaigani
- Haida_Masset
- Tshimshian_Kithatlā
- Tshimshian_Kitunto

## Development Notes

- Frontend uses Vite for fast HMR and building
- Python scripts use modular architecture - each stage can run independently
- All Python scripts include comprehensive error handling and logging
- Fine-tuning jobs are monitored with 60-second polling intervals
- Generated data is stored in JSONL format for OpenAI compatibility