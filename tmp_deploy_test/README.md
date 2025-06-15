---
title: "ft-model-123"
emoji: "💬"
colorFrom: "blue"
colorTo: "indigo"
sdk: "gradio"
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---
# AI Chat Interface

A Gradio-based chat interface for interacting with a fine-tuned OpenAI model.

## Model Information
- Model ID: `ft-model-123`
- Deployed on: 2025-06-15 15:49:46

## Setup
1. Add your `OPENAI_API_KEY` as a secret in the Space settings
2. The app will automatically use the fine-tuned model

## Local Development
```bash
# Clone this repository
git clone https://huggingface.co/spaces/testuser/my-space
cd my-space

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```
