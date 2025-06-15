
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import wandb

# Set up logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class OpenAIFineTuner:
    def __init__(self, dialect: str):
        """
        Initialize the OpenAI fine-tuner with API key and file paths.
        
        Args:
            dialect: Name of the dialect being fine-tuned (e.g., 'Thlinkit_Skutkwan')
        """
        # Load environment variables
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.dialect = dialect
        self.model = "gpt-4.1-2025-04-14"
        logger.info(f"OpenAI client initialized successfully for {dialect}")
        logger.info(f"Using model: {self.model}")
        
        # Define file paths using absolute paths
        self.base_dir = Path(__file__).parent.parent
        self.train_file = self.base_dir / "Output" / f"finetune_qa_{dialect}_train.jsonl"
        self.valid_file = self.base_dir / "Output" / f"finetune_qa_{dialect}_valid.jsonl"
        
        # Ensure files exist
        if not self.train_file.exists():
            raise FileNotFoundError(f"Training file not found: {self.train_file}")
        if not self.valid_file.exists():
            raise FileNotFoundError(f"Validation file not found: {self.valid_file}")
        
        logger.info(f"Found training file: {self.train_file}")
        logger.info(f"Found validation file: {self.valid_file}")

    def upload_file(self, file_path: str, purpose: str) -> str:
        """Upload a file to OpenAI and return its file ID."""
        logger.info(f"Uploading {purpose} file: {file_path}")
        
        with open(file_path, 'rb') as file:
            response = self.client.files.create(
                file=file,
                purpose=purpose
            )
        
        logger.info(f"Successfully uploaded {purpose} file. File ID: {response.id}")
        return response.id

    def create_fine_tuning_job(self, training_file_id: str, validation_file_id: str) -> str:
        """Create a fine-tuning job and return its ID."""
        logger.info(f"Creating fine-tuning job for {self.dialect} using {self.model}...")
        
        response = self.client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=self.model,
            hyperparameters={
                "n_epochs": 3,  # Default number of epochs
                "batch_size": 4,  # Default batch size
                "learning_rate_multiplier": 2.0  # Default learning rate multiplier
            }
        )
        
        logger.info(f"Fine-tuning job created successfully. Job ID: {response.id}")
        return response.id

    def update_env_file(self, model_id: str):
        """Update the .env file with the fine-tuned model ID."""
        env_path = Path(__file__).parent.parent / '.env'
        env_content = []
        model_found = False
        
        # Read existing .env file if it exists
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                env_content = f.read().splitlines()
        
        # Update or add FINE_TUNED_MODEL_ID
        model_line = f"FINE_TUNED_MODEL_ID={model_id}"
        
        for i, line in enumerate(env_content):
            if line.startswith("FINE_TUNED_MODEL_ID="):
                env_content[i] = model_line
                model_found = True
                break
        
        if not model_found:
            env_content.append(model_line)
        
        # Write back to .env
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(env_content) + '\n')
        
        logger.info(f"Updated .env file with FINE_TUNED_MODEL_ID={model_id}")

    def monitor_job_progress(self, job_id: str, check_interval: int = 60, wandb_run=None):
        """Monitor the progress of a fine-tuning job."""
        logger.info(f"Starting to monitor fine-tuning job: {job_id}")
        
        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            status = job.status

            # Log detailed status information
            logger.info(f"Status: {status}")
            trained_tokens = getattr(job, "trained_tokens", None)
            training_accuracy = getattr(job, "training_accuracy", None)
            validation_loss = getattr(job, "validation_loss", None)
            if trained_tokens is not None:
                logger.info(f"Trained tokens: {trained_tokens}")
            if training_accuracy is not None:
                logger.info(f"Training accuracy: {training_accuracy}")
            if validation_loss is not None:
                logger.info(f"Validation loss: {validation_loss}")

            # Log to wandb if available
            if wandb_run is not None:
                try:
                    wandb.log({
                        "status": status,
                        "trained_tokens": trained_tokens,
                        "training_accuracy": training_accuracy,
                        "validation_loss": validation_loss,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as wandb_exc:
                    logger.warning(f"W&B logging error: {wandb_exc}")

            if status == "succeeded":
                model_id = getattr(job, "fine_tuned_model", None)
                if model_id:
                    logger.info("Fine-tuning completed successfully!")
                    logger.info(f"Fine-tuned model ID: {model_id}")
                    # Update .env file with the new model ID
                    try:
                        self.update_env_file(model_id)
                    except Exception as e:
                        logger.error(f"Failed to update .env file: {e}")
                    
                    if wandb_run is not None:
                        wandb_run.summary["fine_tuned_model_id"] = model_id
                else:
                    logger.warning("Fine-tuning succeeded but no model ID was returned")
                break
                
            elif status == "failed":
                error_msg = getattr(job, "error", "Unknown error")
                logger.error(f"Fine-tuning failed: {error_msg}")
                if wandb_run is not None:
                    wandb_run.summary["error"] = error_msg
                break
                
            elif status in ["cancelled", "expired"]:
                logger.warning(f"Fine-tuning job {status}")
                if wandb_run is not None:
                    wandb_run.summary["final_status"] = status
                break

            logger.info(f"Waiting {check_interval} seconds before next check...")
            time.sleep(check_interval)

    def run_fine_tuning(self):
        """Run the complete fine-tuning process."""
        wandb_project = f"openai-finetune-{self.dialect}"
        wandb_run = wandb.init(
            project=wandb_project,
            config={
                "model": self.model,
                "dialect": self.dialect,
                "train_file": str(self.train_file),
                "valid_file": str(self.valid_file),
                "n_epochs": 3,
                "batch_size": 4,
                "learning_rate_multiplier": 2.0,
            }
        )
        try:
            # Step 1: Upload files
            logger.info("Step 1/3: Uploading files to OpenAI")
            train_file_id = self.upload_file(str(self.train_file), "fine-tune")
            valid_file_id = self.upload_file(str(self.valid_file), "fine-tune")
            
            # Step 2: Create fine-tuning job
            logger.info("Step 2/3: Creating fine-tuning job")
            job_id = self.create_fine_tuning_job(train_file_id, valid_file_id)
            wandb.config.update({"openai_job_id": job_id})
            
            # Step 3: Monitor progress
            logger.info("Step 3/3: Monitoring fine-tuning progress")
            self.monitor_job_progress(job_id, wandb_run=wandb_run)
            
        except Exception as e:
            logger.error(f"Error during fine-tuning process: {str(e)}")
            raise
        finally:
            wandb.finish()

def process_all_dialects():
    """Process all dialect dictionary files in the Dictionary directory."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dictionary_dir = project_root / "Dictionary"

    for dict_file in sorted(dictionary_dir.glob("*Dictionary.json")):
        dialect = dict_file.stem.replace("Dictionary", "")
        logger.info(f"=== Starting {dialect} Language Model Fine-Tuning ===")
        try:
            tuner = OpenAIFineTuner(dialect)
            tuner.run_fine_tuning()
            logger.info(f"=== Completed fine-tuning for {dialect} ===")
        except Exception as e:
            logger.error(f"Error processing {dialect}: {e}")

def main():
    process_all_dialects()

if __name__ == "__main__":
    main()
