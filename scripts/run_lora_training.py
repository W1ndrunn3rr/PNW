import os
import json
import shutil
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig


SCRIPTS = {
    "dreambooth": "diffusers/examples/dreambooth/train_dreambooth_lora.py",
    "kandinsky_decoder": "diffusers/examples/kandinsky2_2/text_to_image/train_text_to_image_lora_decoder.py",
    "kandinsky_prior": "diffusers/examples/kandinsky2_2/text_to_image/train_text_to_image_lora_prior.py",
    "wuerstchen": "diffusers/examples/wuerstchen/text_to_image/train_text_to_image_lora_prior.py",
}

PRETRAINED = {
    "dreambooth": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "kandinsky_decoder": "kandinsky-community/kandinsky-2-2-decoder",
    "kandinsky_prior": "kandinsky-community/kandinsky-2-2-prior",
    "wuerstchen": "warp-ai/wuerstchen-prior",
}


def prepare_hf_format(data_root: Path, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    metadata = []
    for class_dir in sorted(data_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.glob("*.png")):
            txt_path = img_path.with_suffix(".txt")
            if not txt_path.exists():
                continue
            dest = f"{class_dir.name}_{img_path.name}"
            shutil.copy(img_path, target / dest)
            metadata.append({"file_name": dest, "text": txt_path.read_text().strip()})
    with open(target / "metadata.jsonl", "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")
    print(f"Prepared {len(metadata)} samples at {target}")
    return target


def base_args(model: str, data_dir: Path, out_dir: Path, cfg: DictConfig) -> list[str]:
    args = [
        f"--pretrained_model_name_or_path={PRETRAINED[model]}",
        f"--output_dir={out_dir}/{model}",
        f"--resolution={cfg.resolution}",
        f"--train_batch_size={cfg.lora.batch_size}",
        f"--gradient_accumulation_steps={cfg.lora.grad_accum}",
        f"--learning_rate={cfg.lora.lr}",
        f"--rank={cfg.lora.rank}",
        f"--max_train_steps={cfg.lora.max_steps}",
        f"--checkpointing_steps={cfg.lora.checkpoint_steps}",
        "--lr_scheduler=cosine",
        "--lr_warmup_steps=0",
        "--report_to=wandb",
    ]
    if model == "dreambooth":
        args.append(f"--instance_data_dir={data_dir}")
        prompt = next(data_dir.rglob("*.txt")).read_text().strip()
        args.append(f"--instance_prompt={prompt}")
    else:
        args.append(f"--train_data_dir={data_dir}")
    if cfg.hub.push:
        args += ["--push_to_hub", f"--hub_model_id={cfg.hub.prefix}-{model}"]
    return args


def launch(model: str, data_root: Path, hf_data: Path, out_dir: Path, cfg: DictConfig):
    precision = "bf16" if model == "wuerstchen" else "fp16"
    data_dir = data_root if model == "dreambooth" else hf_data
    cmd = [
        "accelerate",
        "launch",
        f"--mixed_precision={precision}",
        SCRIPTS[model],
        *base_args(model, data_dir, out_dir, cfg),
    ]
    print(f"\n$ {' '.join(cmd)}\n")
    env = os.environ.copy()
    env["WANDB_PROJECT"] = cfg.wandb.project
    if cfg.hub.token:
        env["HF_TOKEN"] = cfg.hub.token
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{model} failed with code {result.returncode}")


@hydra.main(config_path="conf", config_name="train_lora", version_base=None)
def main(cfg: DictConfig):
    data_root = Path(cfg.data_root)
    out_dir = Path(cfg.output_dir)
    hf_data = out_dir / "_hf_format"

    prepare_hf_format(data_root, hf_data)

    for model in cfg.models:
        print(f"\n{'=' * 60}\n{model.upper()}\n{'=' * 60}")
        launch(model, data_root, hf_data, out_dir, cfg)

    print("\nAll done.")


if __name__ == "__main__":
    main()
