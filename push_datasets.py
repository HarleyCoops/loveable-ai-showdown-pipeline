from huggingface_hub import HfApi
import os

# Initialize the Hugging Face API
api = HfApi()

# Your Hugging Face username and dataset repository name
username = "your-username"  # Replace with your Hugging Face username
dataset_repo = "your-dataset-repo"  # Replace with your dataset repository name

# Path to your datasets
dataset_dir = "Dictionary"

# Push each dataset file
for filename in os.listdir(dataset_dir):
    if filename.endswith(".jsonl") or filename.endswith(".json"):
        file_path = os.path.join(dataset_dir, filename)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=f"{username}/{dataset_repo}",
            repo_type="dataset"
        )
        print(f"Uploaded {filename} to {username}/{dataset_repo}") 