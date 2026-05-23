import modal
import subprocess
import os

app = modal.App("unistainnet-training")

# 1. Define the Cloud Environment (installs all dependencies)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("unzip") # Required for dataset extraction
    .pip_install(
        "torch", 
        "torchvision", 
        "pytorch-lightning", 
        "timm", 
        "lpips", 
        "torchmetrics", 
        "wandb", 
        "huggingface_hub"
    )
    # Sync your local UNIStainNet code folder to the cloud container
    .add_local_dir(".", remote_path="/root/UNIStainNet") 
)

# 2. Define Persistent Storage (survives after GPU turns off)
vol = modal.Volume.from_name("unistainnet-data-vol", create_if_missing=True)

# -------------------------------------------------------------------------
# SETUP SCRIPT: Run this ONCE to download the dataset to your Modal Volume
# -------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")], # Needs HF_TOKEN
    timeout=7200 # 2 hours max
)
def setup_dataset():
    """Downloads and unzips the dataset directly into the Modal Volume."""
    os.makedirs("/data/MIST", exist_ok=True)
    os.chdir("/data")
    
    print("Downloading dataset from Hugging Face to Modal Volume...")
    cmd = [
        "hf", "download", 
        "asserelzeki/destained-histopathology-data", 
        "--repo-type", "dataset", 
        "--local-dir", "."
    ]
    
    token = os.environ.get("HF_TOKEN")
    if token:
        cmd.extend(["--token", token])
    else:
        print("WARNING: HF_TOKEN not found in secrets! Download may fail if dataset is private.")
        
    subprocess.run(cmd, check=True)

    print("Unzipping datasets...")
    stains = ["HER2", "ER", "Ki67", "PR"]
    for stain in stains:
        zip_path = f"{stain}-Destained.zip"
        if os.path.exists(zip_path):
            print(f"Unzipping {stain}...")
            subprocess.run(["unzip", "-q", "-o", zip_path, "-d", "MIST/"])
            
            # Rename the folder like in your bash script
            old_dir = f"MIST/{stain}-Destained"
            new_dir = f"MIST/{stain}"
            if os.path.exists(old_dir):
                os.rename(old_dir, new_dir)
        else:
            print(f"Skipping {stain}, zip not found.")
            
    print("Dataset setup complete! Files are saved in the Modal Volume.")


# -------------------------------------------------------------------------
# TRAINING SCRIPT: Runs train_mist.py on an A100 GPU
# -------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100", # Can change to A10G if you want to save credits
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret")
    ],
    timeout=86400, # 24 hours max
)
def train_model(stains: str, batch_size: int, wandb_name: str):
    os.chdir("/root/UNIStainNet")
    
    # We use PYTHONPATH=. so python can find the src/ folder
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    cmd = [
        "python", "scripts/train/train_mist.py",
        "--data_dir", "/data/MIST",
        "--ckpt_dir", "/data/checkpoints",
        "--stains", *stains.split(),
        "--batch_size", str(batch_size),
        "--wandb_name", wandb_name
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


# -------------------------------------------------------------------------
# LOCAL ENTRYPOINTS: These are what you type in your terminal to launch
# -------------------------------------------------------------------------
@app.local_entrypoint()
def setup():
    """Run: modal run scripts/train/modal_wrapper.py::setup"""
    setup_dataset.remote()

@app.local_entrypoint()
def train(stains: str = "ER PR", batch_size: int = 8, wandb_name: str = "ccpl_er_pr"):
    """Run: modal run scripts/train/modal_wrapper.py::train --stains 'ER PR'"""
    train_model.spawn(stains, batch_size, wandb_name)
