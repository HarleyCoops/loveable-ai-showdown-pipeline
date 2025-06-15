
import gradio as gr
import openai
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration - will be set by deployment script
MODEL = "{{MODEL_NAME}}"  # This will be replaced by the deployment script
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def initialize_openai_client():
    """Initialize OpenAI client with error handling"""
    try:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {str(e)}")
        raise

def chat_with_model(message: str, history: list) -> str:
    """Chat with the fine-tuned model"""
    try:
        logger.info(f"Received message at {datetime.now()}: {message}")
        
        # Format conversation history
        messages = []
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": message})
        
        # Initialize client and make API call
        client = initialize_openai_client()
        logger.info(f"Sending request to model: {MODEL}")
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
        )
        
        assistant_message = response.choices[0].message.content
        logger.info("Received response from model")
        return assistant_message
    except Exception as e:
        error_msg = f"Error processing your request: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Create the Gradio interface
def create_app(title, description, examples=None):
    """Create and return the Gradio chat interface"""
    if examples is None:
        examples = [
            "Hello! How are you today?",
            "What kind of questions can you help me with?",
            "Tell me about yourself"
        ]
    
    return gr.ChatInterface(
        fn=chat_with_model,
        title=title,
        description=description,
        examples=examples,
        theme=gr.themes.Soft()
    )

# For local testing
if __name__ == "__main__":
    # These values will be replaced by the deployment script
    app = create_app(
        title="💬 Chat with AI",
        description="Welcome to the AI Chat Interface!"
    )
    app.launch()
