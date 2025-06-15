#!/usr/bin/env python3
"""
HFSpacesDeployment - Automated deployment of fine-tuned models to Hugging Face Spaces.

This script automates the process of deploying a Gradio chat interface to Hugging Face Spaces
for a newly fine-tuned OpenAI model. It will:
1. Retrieve the latest fine-tuned model from OpenAI
2. Create a new Hugging Face Space
3. Deploy the chat interface with the model

Requirements:
- Python 3.8+
- OpenAI Python package
- Hugging Face Hub Python package
- GitPython
"""

import os
import sys
import argparse
import logging
import openai
from pathlib import Path
from typing import Optional
import tempfile
import shutil
from datetime import datetime
from huggingface_hub import HfApi, Repository, create_repo
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
APP_TEMPLATE_PATH = Path("HFSpacesApp/app_template.py")
REQUIREMENTS_TEMPLATE = """gradio>=4.0.0
openai>=1.0.0
python-dotenv>=1.0.0
"""

class HFSpacesDeployer:
    def __init__(self, hf_token: Optional[str] = None, openai_api_key: Optional[str] = None):
        # Try to get HF token from args, then from either HF_TOKEN or HUGGINGFACE_API_KEY env vars
        self.hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.hf_token:
            raise ValueError(
                "Hugging Face token not provided and not found in environment variables. "
                "Set HF_TOKEN or HUGGINGFACE_API_KEY environment variable or use --hf-token"
            )
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
        self.hf_api = HfApi(token=self.hf_token)
        openai.api_key = self.openai_api_key
        self.load_env()
    
    def load_env(self):
        """Load environment variables from .env file if it exists."""
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key] = value.strip('\"\'')

    def get_model_id(self, model_id: Optional[str] = None) -> str:
        if model_id:
            logger.info(f"Using provided model ID: {model_id}")
            return model_id
            
        env_model_id = os.getenv("FINE_TUNED_MODEL_ID")
        if env_model_id:
            logger.info(f"Using model ID from .env: {env_model_id}")
            return env_model_id
            
        logger.info("No model ID provided, attempting to find the latest fine-tuned model...")
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            models = client.fine_tuning.jobs.list(limit=1)
            if not models.data:
                raise ValueError("No fine-tuned models found")
            latest_model = models.data[0]
            if latest_model.status != "succeeded":
                raise ValueError(f"Latest model training job is not complete. Status: {latest_model.status}")
            model_id = latest_model.fine_tuned_model
            if not model_id:
                raise ValueError("Model ID not available in the fine-tuning job")
            logger.info(f"Found latest fine-tuned model: {model_id}")
            return model_id
        except Exception as e:
            error_msg = "Failed to determine model ID. Please provide a model ID using --model-id or run the fine-tuning script first."
            logger.error(f"{error_msg} Error: {str(e)}")
            raise ValueError(error_msg) from e

    def _prepare_app_files(self, model_id: str, space_name: str, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        template = APP_TEMPLATE_PATH.read_text(encoding='utf-8')
        processed_template = template.replace("{{MODEL_NAME}}", model_id)
        (output_dir / "app.py").write_text(processed_template, encoding='utf-8')
        (output_dir / "requirements.txt").write_text(REQUIREMENTS_TEMPLATE, encoding='utf-8')
        username = self.hf_api.whoami()['name']
        deploy_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Patch: Provide actual metadata, not placeholders
        yaml_block = (
            f"""---
"""
            f"title: \"{model_id}\"\n"
            f"emoji: \"💬\"\n"
            f"colorFrom: \"blue\"\n"
            f"colorTo: \"indigo\"\n"
            f"sdk: gradio\n"
            f"sdk_version: \"4.0.0\"\n"
            f"app_file: app.py\n"
            f"pinned: false\n"
            f"---\n"
        )
        # Note: No blank line after the YAML, markdown starts immediately.
        readme_content = f"""{yaml_block}# AI Chat Interface

A Gradio-based chat interface for interacting with a fine-tuned OpenAI model.

## Model Information
- Model ID: `{model_id}`
- Deployed on: {deploy_date}

## Setup
1. Add your `OPENAI_API_KEY` as a secret in the Space settings
2. The app will automatically use the fine-tuned model

## Local Development
```bash
# Clone this repository
git clone https://huggingface.co/spaces/{username}/{space_name}
cd {space_name}

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```
"""
        (output_dir / "README.md").write_text(readme_content, encoding='utf-8')
        logger.info(f"Prepared app files in {output_dir}")

    def deploy_to_hf_spaces(
        self, 
        space_name: str, 
        model_id: Optional[str] = None,
        organization: Optional[str] = None,
        private: bool = True,
        skip_if_exists: bool = True,
        verbose: bool = False
    ) -> str:
        space_name = space_name.lower().replace(" ", "-")
        if verbose:
            logger.setLevel(logging.DEBUG)
            logger.debug(f"Starting deployment with space_name={space_name}, organization={organization}")
        try:
            model_id = self.get_model_id(model_id)
            logger.info(f"Using model ID: {model_id}")
        except Exception as e:
            logger.error(f"Failed to get model ID: {str(e)}")
            raise
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger.debug(f"Created temporary directory: {temp_path}")
            try:
                logger.info("Preparing application files...")
                self._prepare_app_files(model_id, space_name, temp_path)
                username = self.hf_api.whoami()['name']
                repo_owner = organization or username
                repo_url = f"https://huggingface.co/spaces/{repo_owner}/{space_name}"
                logger.info(f"Checking if space {repo_owner}/{space_name} already exists...")
                space_exists = False
                if skip_if_exists:
                    try:
                        self.hf_api.repo_info(repo_id=f"{repo_owner}/{space_name}", repo_type="space")
                        space_exists = True
                        logger.info(f"Space {repo_owner}/{space_name} already exists. Skipping deployment as skip_if_exists=True")
                        return repo_url
                    except Exception as e:
                        if "404" in str(e):
                            logger.info(f"Space {repo_owner}/{space_name} does not exist, will create new space")
                        else:
                            logger.warning(f"Error checking if space exists: {str(e)}")
                        space_exists = False

                if not space_exists:
                    logger.info(f"Creating new space: {repo_owner}/{space_name}")
                    create_repo(
                        repo_id=f"{repo_owner}/{space_name}",
                        repo_type="space",
                        space_sdk="gradio",
                        token=self.hf_token,
                        private=private,
                        exist_ok=True
                    )
                    logger.info(f"Created new space: {repo_url}")

                logger.info("Cloning the repository...")
                repo = Repository(
                    local_dir=str(temp_path / "repo"),
                    clone_from=f"https://huggingface.co/spaces/{repo_owner}/{space_name}",
                    use_auth_token=self.hf_token,
                    skip_lfs_files=True
                )

                logger.info("Copying application files...")
                repo_dir = Path(repo.local_dir)
                for file in temp_path.glob("*"):
                    if file.name != "repo":
                        dest = repo_dir / file.name
                        if file.is_dir():
                            if dest.exists():
                                shutil.rmtree(dest)
                            shutil.copytree(file, dest)
                        else:
                            shutil.copy2(file, dest)

                logger.info("Committing and pushing changes...")
                repo.git_add(auto_lfs_track=True)
                repo.git_commit(f"Deploy chat interface for model {model_id}")
                repo.git_push(blocking=True)

                logger.info(f"Successfully deployed to {repo_url}")
                return repo_url

            except Exception as e:
                logger.error(f"Error during deployment: {str(e)}")
                if verbose:
                    logger.exception("Full traceback:")
                raise

def parse_args():
    parser = argparse.ArgumentParser(description="Deploy a fine-tuned model to Hugging Face Spaces")
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face authentication token (default: HF_TOKEN or HUGGINGFACE_API_KEY environment variable)"
    )
    parser.add_argument(
        "--openai-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (default: OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--dialect-name",
        type=str,
        default="thlinkit-skutkwan",
        help="Name of the dialect (default: thlinkit-skutkwan, will be used to generate space name)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        help="OpenAI model ID to use (default: latest fine-tuned model)"
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="Organization to create the space under (default: your username)"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the space public (default: private)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    logger.info("Starting Hugging Face Spaces deployment...")
    space_name = args.dialect_name.lower().replace(" ", "-")
    logger.info(f"Using dialect name: {args.dialect_name}")
    logger.info(f"Generated space name: {space_name}")
    logger.debug("\n=== Environment Variables ===")
    for key, value in os.environ.items():
        if any(k in key.upper() for k in ['HF_', 'HUGGINGFACE', 'OPENAI']):
            logger.debug(f"{key}: {'*' * 8 if 'TOKEN' in key or 'KEY' in key else value}")
    logger.debug("===========================\n")
    hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if not hf_token:
        error_msg = (
            "Hugging Face token is required. "
            "Set HF_TOKEN or HUGGINGFACE_API_KEY environment variable or use --hf-token\n"
            "Current environment keys: " + ", ".join(k for k in os.environ.keys() if 'HF_' in k or 'HUGGING' in k)
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    args.hf_token = hf_token
    try:
        logger.info("Initializing deployment...")
        deployer = HFSpacesDeployer(
            hf_token=args.hf_token,
            openai_api_key=args.openai_key
        )
        logger.info(f"Starting deployment to space: {space_name}")
        space_url = deployer.deploy_to_hf_spaces(
            space_name=space_name,
            model_id=args.model_id,
            organization=args.organization,
            private=not args.public,
            verbose=args.verbose
        )
        print(f"\n✅ Successfully deployed to: {space_url}")
        print(f"   - Make sure to add your OPENAI_API_KEY as a secret in the Space settings")
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        if args.verbose:
            logger.exception("Full traceback:")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if '--verbose' in sys.argv or '-v' in sys.argv:
            logger.exception("Full traceback:")
        sys.exit(1)
