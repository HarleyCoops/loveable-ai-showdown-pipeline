# Mechanistic Interpretability of Linguistic Patterns in Low-Resource Languages

![Map showing historical migration](./Public/FullDawsonMap.jpg)

## Research Hypothesis

This project explores the theory from Google DeepMind labs that linguistic patterns can be observed through mechanistic interpretability. By training models on closely related low-resource language variants (Haida, Tlingit, and Tshimshian dialects), we can use neuron activation analysis to understand how language models internally represent and process linguistic relationships.

## The Vision

Using the 1884 Dawson dictionary as our foundation, this dataset provides a unique opportunity to:

1. **Observe Linguistic DNA**: Track how neural networks encode relationships between related languages
2. **Visualize Translation Reasoning**: Use mechanistic interpretability tools to "watch" neurons fire during translation tasks
3. **Map Language Evolution**: Understand how models represent dialectical variations within language families
4. **Decode Internal Representations**: Identify which neurons activate for specific linguistic features

## Research Methodology

### Phase 1: Data Generation (Current)
- Extract entries from historical dictionaries
- Generate synthetic QA pairs using advanced language models
- Create fine-tuning datasets for each dialect variant

### Phase 2: Open Source Model Training (Planned)
- Train smaller, interpretable models (e.g., GPT-2, BERT variants)
- Fine-tune on our dialect-specific datasets
- Maintain full visibility into model architectures

### Phase 3: Mechanistic Analysis (Future)
- Apply Google DeepMind's interpretability tools
- Monitor neuron activations during inference
- Map which neurons encode specific linguistic features
- Identify cross-dialect pattern recognition

## Why This Dataset?

The closely related nature of these Pacific Northwest languages provides an ideal testbed for mechanistic interpretability because:

- **Controlled Variation**: Dialects share core structures with systematic differences
- **Historical Documentation**: The 1884 Dawson dictionary provides authenticated linguistic data
- **Cultural Preservation**: Supports First Nations communities while advancing AI research
- **Scientific Value**: Reveals how neural networks encode linguistic relationships



 ## Quickstart: Run the Full Pipeline

 Simply run the following command to execute the entire pipeline end-to-end (data extraction, synthetic data generation, format conversion, and model fine-tuning) for all dialects:

 ```bash
 python run_full_pipeline.py
 ```

 ## Manual Steps

 If you want more control or need to customize individual parts, you can run each step separately:

 1. **Generate synthetic question–answer pairs**
    ```bash
    python Scripts/openAI_bilingual_qa_generator.py
    ```

 2. **Convert to fine-tuning format & split data**  
    Replace `<Dialect>` with the name of the dialect (e.g., `Thlinkit_Skutkwan`):
    ```bash
    python Scripts/convert_qa_to_finetune.py \
      --dialect <Dialect> \
      --input Dictionary/synthetic_qa_<Dialect>_openai.jsonl \
      --output Output/finetune_qa_<Dialect>
    ```

 3. **Launch the OpenAI fine-tuning jobs**
    ```bash
    python Scripts/openai_finetune.py
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