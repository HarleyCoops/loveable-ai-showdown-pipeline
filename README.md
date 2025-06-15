# From Dictionary to Training: The Low-Resource Language Trainer

![Map showing historical migration](./Public/FullDawsonMap.jpg)

 What if you could take any dictionary of any low-resource language, process the text, extract meaningful context, generate synthetic data, then fine-tune a working model all in one step?

 Using the 1884 Dawson map and dictionary, I have constructed this test case with a complete pipeline of extraction, processing, inference, and training datasets that will allow any First Nations community to begin training their own models toward fluency.

 What is particularly interesting about this dataset is that it contains variations of languages that will allow techniques pioneered by Google DeepMind and Anthropic to actually peer into the activated neurons of the LLM and see how it is reasoning through translation.

 While this model was trained with OpenAI, future training on smaller, open source models might also yield elements of linguistic DNA that tell us how we all might be reasoning with language.



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