import os
import subprocess
from accelerate.utils import write_basic_config
import wandb
import torch


def setup_environment():
    write_basic_config(mixed_precision="fp16")

    if os.getenv("WANDB_API_KEY"):
        wandb.login(key=os.getenv("WANDB_API_KEY"))

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


if __name__ == "__main__":
    setup_environment()
