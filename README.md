# From Dictionary to OpenAI: A Low-Resource Language Trainer

![Map showing historical migration](Public/FullDawsonMap.jpg)

A comprehensive pipeline for preserving and revitalizing Canadian First Nations languages through advanced NLP techniques, leveraging historical dictionaries, synthetic data generation, and modern language models.

## The Story Hidden in Neural Networks

Twenty thousand years ago, the first humans crossed the Beringia land bridge into North America. As they moved south along the Pacific coast, their common tongue began the slow drift that would eventually yield hundreds of languages. Ten thousand years later, as an inland corridor opened through retreating ice sheets, a second group split off, bringing their speech into what is now Alberta. This project explores that ancient divergence through modern AI, training separate language models on coastal and interior dialects. These models are not meant to be immediately fluent—though they can become so with enough data. Instead, the focus is on how linguistic patterns differ and how meaning drifts across millennia.


## Technical Innovation

Our approach uses Canonical Correlation Analysis (CCA), logit lens techniques and LoRA-based fine-tuning. These tools allow causal interventions on the models to see where dialect-specific meaning is stored while keeping resource requirements low for endangered languages.

## Project Narrative

This project began with a remarkable historical resource: an 1884 dictionary documenting several Canadian First Nations languages. Our mission is to breathe new life into these linguistic treasures by creating a robust pipeline that transforms static dictionary entries into interactive language learning tools.

The dictionary contains five distinct dialects:
- Haida (Kaigani and Masset variants)
- Thlinkit (Skutkwan variant)
- Tshimshian (Kithatlā and Kitunto variants)

## Technical Approach

Our pipeline follows these key steps:

1. **Data Extraction & Vectorization**
   - Process historical dictionary entries into structured JSON format
   - Establish ground truth through careful validation of translations and usage examples

2. **Synthetic Data Generation**
   - Use OpenAI's advanced language models to generate diverse, naturalistic Q&A pairs
   - Ensure synthetic data maintains strict adherence to original dictionary meanings
   - Create contextual examples that demonstrate practical usage of each term

3. **Model Fine-tuning**
   - Prepare training/validation splits from both original and synthetic data
   - Fine-tune models using OpenAI's API with Weights & Biases integration
   - Implement Hugging Face integration for model sharing and versioning

4. **Evaluation & Iteration**
   - Monitor training metrics through W&B dashboards
   - Validate model outputs against ground truth
   - Continuously improve through iterative refinement

### Key Features

- **Historical Dictionary Processing**: Careful handling of 19th-century linguistic data
- **Synthetic Data Pipeline**: Automated generation of high-quality training examples
- **Modern NLP Integration**: Leveraging OpenAI's latest models for fine-tuning
- **Training Transparency**: Comprehensive monitoring via Weights & Biases
- **Open Source**: All code and models available on Hugging Face for community use

## Project Structure

```
.
├── Dictionary/              # Source dictionaries and generated datasets
├── Scripts/                 # Processing and training scripts
├── SourcePDFs/             # Original source materials
├── prompts/                # LLM prompts for synthetic data generation
└── instructions.txt        # Project documentation and goals
```

## Setup

### Prerequisites

- Python 3.9+
- Git
- OpenAI API key
- Weights & Biases account (for experiment tracking)
- Hugging Face account (for model sharing)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HarleyCoops/LoveableAIShowdown.git
   cd LoveableAIShowdown
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   WANDB_API_KEY=your_wandb_api_key
   HUGGINGFACE_TOKEN=your_hf_token
   ```

## Usage

### 1. Data Processing Pipeline

1. **Convert Raw Dictionary Data**
   ```bash
   python Scripts/convert_data_format.py --input Dictionary/raw/ --output Output/processed/
   ```

2. **Generate Synthetic Q&A Pairs**
   ```bash
   python Scripts/OpenAI_bilingual_qa_generator.py --input Output/processed/ --output Output/synthetic_qa/
   ```

3. **Convert to Fine-tuning Format**
   ```bash
   python Scripts/convert_qa_to_finetune.py --input Output/synthetic_qa/ --output Output/finetune/
   ```

### 2. Model Fine-tuning

1. **Set up Fine-tuning**
   ```bash
   python Scripts/finetunesetup.py --config configs/training_config.yaml
   ```

2. **Run Fine-tuning with W&B Integration**
   ```bash
   python Scripts/openai_finetune.py --input Output/finetune/ --output Output/models/
   ```

3. **Monitor Training**
   - Training progress can be monitored in real-time via W&B dashboard
   - Checkpoints are automatically saved to the specified output directory

### 3. Model Evaluation

1. **Run Evaluation Script**
   ```bash
   python Scripts/evaluate_model.py --model Output/models/checkpoint-latest --test_data Output/finetune/test.jsonl
   ```

2. **View Results**
   - Evaluation metrics are logged to W&B
   - Detailed reports are saved in `Output/evaluation/`

### 4. Automated Deployment to Hugging Face Spaces

After fine-tuning your model, you can easily deploy it as a chat interface on Hugging Face Spaces:

1. **Install deployment dependencies**:
   ```bash
   pip install -r requirements-deploy.txt
   ```

2. **Set up authentication**:
   ```bash
   export HF_TOKEN=your_huggingface_token
   export OPENAI_API_KEY=your_openai_api_key
   ```

3. **Deploy your model**:
   ```bash
   python HFSpacesDeployment.py --space-name "your-language-assistant" --public
   ```

   Options:
   - `--space-name`: Name for your Hugging Face Space (required)
   - `--public`: Make the space public (optional)
   - `--organization`: Deploy to an organization (optional)
   - `--model-id`: Specify a model ID (defaults to latest fine-tuned model)

4. **After deployment**:
   - Visit your Hugging Face Spaces dashboard
   - Go to your new Space's settings
   - Add your `OPENAI_API_KEY` as a repository secret
   - The app will automatically use your fine-tuned model

Your chat interface will be available at:  
`https://huggingface.co/spaces/your-username/your-space-name`

## Technical Implementation

### Data Pipeline

1. **Dictionary Processing**
   - Raw dictionary entries are parsed and cleaned
   - Entries are converted to a structured JSON format
   - Metadata and translations are validated for consistency
   - Data is split into train/validation/test sets (80/10/10)

2. **Synthetic Data Generation**
   - Uses OpenAI's API to generate diverse Q&A pairs
   - Implements strict validation against original dictionary meanings
   - Generates contextual examples for practical usage
   - Applies data augmentation techniques for robustness

3. **Model Architecture**
   - Based on OpenAI's fine-tuning capabilities
   - Implements transfer learning from pre-trained models
   - Supports both few-shot and zero-shot learning
   - Includes safety and bias mitigation layers

4. **Training Infrastructure**
   - Distributed training support
   - Mixed precision training
   - Gradient checkpointing for large models
   - Automatic mixed precision (AMP) support

### Monitoring & Evaluation

- Real-time metrics tracking via W&B
- Automated model checkpointing
- Performance benchmarking against baseline models
- Bias and fairness evaluation
- Interactive error analysis tools


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




## Contact

For questions or feedback, please open an issue on GitHub or contact the maintainers.