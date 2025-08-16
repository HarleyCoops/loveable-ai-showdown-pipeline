# Pacific Northwest Indigenous Language AI Preservation Pipeline

![Map showing historical migration patterns of Pacific Northwest First Nations](./Public/FullDawsonMap.jpg)

> **Transforming endangered language dictionaries into living, interactive AI systems that preserve and teach indigenous languages for future generations.**

## Project Significance

This repository contains a **production-ready pipeline** that represents a breakthrough in preserving critically endangered Pacific Northwest First Nations languages. By converting static dictionary entries from the 1884 Dawson linguistic survey into fine-tuned AI models, we're creating the first comprehensive AI-powered preservation system for Haida, Tlingit, and Tshimshian language families.

### Critical Impact Areas

**Cultural Preservation**: Only ~20 fluent Haida speakers remain worldwide. This pipeline transforms historical documentation into accessible, interactive AI tutors that can help revitalize these languages.

**Scientific Innovation**: Implements Google DeepMind's mechanistic interpretability theories to understand how AI models encode and process linguistic relationships between related dialects.

**Global Template**: Provides a replicable framework for preserving any of the world's 7,000+ languages, 40% of which are currently endangered.

**Community Accessibility**: Deploys models as free, public tools on Hugging Face Spaces, ensuring indigenous communities and linguists have direct access to these resources.

## Pipeline Architecture

This sophisticated document processing pipeline automates the complete journey from dictionary to deployed AI:

```
Dictionary JSON → QA Generation (500+ pairs) → Fine-tuning (GPT-4) → Deployment (HuggingFace) → Public Access
```

### Supported Languages and Dialects
- **Haida**: Kaigani and Masset dialects (~1,141 dictionary entries each)
- **Tlingit**: Skutkwan dialect
- **Tshimshian**: Kithatlā and Kitunto dialects

## Technical Innovation

### Advanced Capabilities
1. **Synthetic Data Amplification**: Transforms limited dictionary entries into rich QA datasets using state-of-the-art reasoning models
2. **Multi-Model Architecture**: Extensible framework supporting OpenAI, Gemma, and future open-source models
3. **Automated Deployment**: Single-command deployment to interactive web applications
4. **Interpretability-First Design**: Built for neuron-level analysis of linguistic encoding patterns

## Research Foundation

### Mechanistic Interpretability Approach

This project explores cutting-edge theories from Google DeepMind labs regarding how linguistic patterns can be observed through mechanistic interpretability. By training models on closely related low-resource language variants, we can use neuron activation analysis to understand how language models internally represent and process linguistic relationships.

### Research Phases

**Phase 1: Data Generation** (Current)
- Extract and validate entries from historical dictionaries
- Generate synthetic QA pairs using advanced language models
- Create fine-tuning datasets for each dialect variant

**Phase 2: Model Training** (In Progress)
- Fine-tune GPT-4 class models on dialect-specific datasets
- Implement comprehensive evaluation metrics
- Track training progress with Weights & Biases

**Phase 3: Mechanistic Analysis** (Planned)
- Apply interpretability tools to understand model representations
- Monitor neuron activations during translation tasks
- Map cross-dialect pattern recognition
- Identify which neural pathways encode specific linguistic features

## Why This Dataset Matters

The closely related nature of these Pacific Northwest languages provides an ideal testbed for advancing AI understanding:

- **Controlled Variation**: Dialects share core structures with systematic differences, allowing precise analysis of how models encode variations
- **Historical Authentication**: The 1884 Dawson dictionary provides rigorously documented linguistic data from native speakers
- **Preservation Urgency**: With fewer than 100 combined fluent speakers across all dialects, this work is time-critical
- **Scientific Advancement**: Reveals fundamental insights into how neural networks encode and process human language



## Quickstart: Run the Full Pipeline

The new modular package exposes a single `main.py` CLI. Running the full
pipeline is now as simple as:

```bash
python main.py generate_qa --dialect-name Thlinkit_Skutkwan \
    --input Dictionary/Thlinkit_SkutkwanDictionary.json \
    --output Dictionary/synthetic_qa_Thlinkit_Skutkwan_openai.jsonl
python main.py convert --dialect Thlinkit_Skutkwan \
    --input Dictionary/synthetic_qa_Thlinkit_Skutkwan_openai.jsonl \
    --output Output/finetune_qa_Thlinkit_Skutkwan
python main.py finetune_openai --dialect Thlinkit_Skutkwan
```

 ## Manual Steps

 If you want more control or need to customize individual parts, you can run each step separately:

1. **Generate synthetic question–answer pairs**
   ```bash
   python main.py generate_qa --dialect-name <Dialect> \
       --input Dictionary/<Dialect>Dictionary.json \
       --output Dictionary/synthetic_qa_<Dialect>_openai.jsonl
   ```

 2. **Convert to fine-tuning format & split data**  
    Replace `<Dialect>` with the name of the dialect (e.g., `Thlinkit_Skutkwan`):
   ```bash
   python main.py convert \
     --dialect <Dialect> \
     --input Dictionary/synthetic_qa_<Dialect>_openai.jsonl \
     --output Output/finetune_qa_<Dialect>
   ```

 3. **Launch the OpenAI fine-tuning jobs**
   ```bash
   python main.py finetune_openai --dialect <Dialect>
   ```

 Outputs and logs for each step can be found in the `Output/` folder.

 ## Optional & Cleanup

 - **Process a single dialect only**: pass `--dialect` to step 2 and edit `run_full_pipeline.py` if needed.
 - **Tidy up intermediate files**:
   ```bash
   rm -rf Output/synthetic_qa_* Output/finetune_qa_*
   ```
 - **Clear old experiment runs** (Weights & Biases):
   ```bash
   wandb gc
   ```



 ## License

 This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

 ## Contact

 For questions or feedback, please open an issue on GitHub or contact the maintainers.

## Deploy to Hugging Face Spaces (per dialect)

To deploy a Gradio chat app for any dialect, copy and run the corresponding command below (ensure your .env is set up with your Hugging Face and OpenAI credentials):

| Dialect Name              | Deployment Command                                                                 |
|--------------------------|----------------------------------------------------------------------------------|
| Thlinkit_Skutkwan        | `python HFSpacesDeployment.py --dialect-name "Thlinkit_Skutkwan" --verbose`        |
| Haida_Kaigani            | `python HFSpacesDeployment.py --dialect-name "Haida_Kaigani" --verbose`            |
| Haida_Masset             | `python HFSpacesDeployment.py --dialect-name "Haida_Masset" --verbose`             |
| Tshimshian_Kithatlā      | `python HFSpacesDeployment.py --dialect-name "Tshimshian_Kithatlā" --verbose`      |
| Tshimshian_Kitunto       | `python HFSpacesDeployment.py --dialect-name "Tshimshian_Kitunto" --verbose`       |

> **Tip:** Add `--public` to the command if you want the Hugging Face Space to be public.