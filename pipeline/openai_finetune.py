"""OpenAI fine-tuning utilities."""
import logging
import os
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
import wandb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpenAIFineTuner:
    def __init__(self, dialect: str, model: str = "gpt-4.1-2025-04-14"):
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError('OPENAI_API_KEY not found in environment variables')
        self.client = OpenAI(api_key=api_key)
        self.dialect = dialect
        self.model = model
        root = Path(__file__).resolve().parents[1]
        self.train_file = root / "Output" / f"finetune_qa_{dialect}_train.jsonl"
        self.valid_file = root / "Output" / f"finetune_qa_{dialect}_valid.jsonl"
        if not self.train_file.exists() or not self.valid_file.exists():
            raise FileNotFoundError('Training or validation file missing')

    def upload_file(self, file_path: str, purpose: str) -> str:
        with open(file_path, 'rb') as f:
            response = self.client.files.create(file=f, purpose=purpose)
        return response.id

    def create_job(self, train_id: str, valid_id: str) -> str:
        response = self.client.fine_tuning.jobs.create(
            training_file=train_id,
            validation_file=valid_id,
            model=self.model,
            hyperparameters={"n_epochs": 3, "batch_size": 4, "learning_rate_multiplier": 2.0},
        )
        return response.id

    def monitor(self, job_id: str):
        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            logger.info(f"Status: {status}")
            if status == "succeeded":
                model_id = getattr(job, "fine_tuned_model", None)
                if model_id:
                    logger.info(f"Fine-tuned model ID: {model_id}")
                break
            elif status in {"failed", "cancelled", "expired"}:
                break
            import time
            time.sleep(60)

    def run(self):
        run = wandb.init(project=f"openai-finetune-{self.dialect}")
        try:
            train_id = self.upload_file(str(self.train_file), "fine-tune")
            valid_id = self.upload_file(str(self.valid_file), "fine-tune")
            job_id = self.create_job(train_id, valid_id)
            self.monitor(job_id)
        finally:
            wandb.finish()

