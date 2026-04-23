set dotenv-load

root := justfile_directory()

default:
    @just --list

setup:
    uv sync --no-dev
    uv run setup.py
    git clone --depth=1 https://github.com/huggingface/diffusers || true
    pip install -e diffusers
    pip install -r diffusers/examples/dreambooth/requirements.txt
    pip install -r diffusers/examples/kandinsky2_2/text_to_image/requirements.txt
    pip install -r diffusers/examples/wuerstchen/text_to_image/requirements.txt
    pip install wandb huggingface_hub
    accelerate config

prepare:
    uv run scripts/prepare_datasets.py

prepare-one dataset:
    uv run scripts/prepare_datasets.py datasets=[{{dataset}}]

train-all:
    uv run scripts/run_lora_training.py

train-all-push:
    uv run scripts/run_lora_training.py hub.push=true hub.token=$HF_TOKEN

train-dreambooth:
    uv run scripts/run_lora_training.py "models=[dreambooth]"

train-kandinsky:
    uv run scripts/run_lora_training.py "models=[kandinsky_decoder,kandinsky_prior]"

train-wuerstchen:
    uv run scripts/run_lora_training.py "models=[wuerstchen]"

pipeline: prepare train-all-push

login-wandb:
    wandb login

login-hf:
    huggingface-cli login

clean:
    rm -rf outputs

clean-all: clean
    rm -rf data diffusers
