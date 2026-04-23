import subprocess
from pathlib import Path

import torch
import wandb
from torch_fidelity import calculate_metrics

MODELS = {
    "flux": "black-forest-labs/FLUX.1-schnell",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sana": "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
}

HORSE_DATASET = "./data/horses"
OUTPUT_DIR = "./stage1"
PROMPT = "a photo of a horse"
NUM_IMAGES = 100
LORA_RANK = 8
TRAIN_STEPS = 500


def train_lora(model_name, model_id):
    out = Path(OUTPUT_DIR) / model_name
    out.mkdir(parents=True, exist_ok=True)

    script = {
        "flux": "diffusers/examples/dreambooth/train_dreambooth_lora_flux.py",
        "sdxl": "diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py",
        "sana": "diffusers/examples/dreambooth/train_dreambooth_lora_sana.py",
    }[model_name]

    cmd = [
        "accelerate",
        "launch",
        script,
        f"--pretrained_model_name_or_path={model_id}",
        f"--instance_data_dir={HORSE_DATASET}",
        f"--instance_prompt={PROMPT}",
        f"--output_dir={out}/lora",
        f"--rank={LORA_RANK}",
        f"--max_train_steps={TRAIN_STEPS}",
        "--train_batch_size=1",
        "--mixed_precision=fp16",
        "--gradient_checkpointing",
        "--report_to=wandb",
    ]

    subprocess.run(cmd, check=True)


def generate_images(model_name, model_id):
    from diffusers import AutoPipelineForText2Image

    lora_path = Path(OUTPUT_DIR) / model_name / "lora"
    gen_path = Path(OUTPUT_DIR) / model_name / "generated"
    gen_path.mkdir(parents=True, exist_ok=True)

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to("cuda")
    pipe.load_lora_weights(str(lora_path))

    for i in range(NUM_IMAGES):
        image = pipe(PROMPT, num_inference_steps=25).images[0]
        image.save(gen_path / f"{i:04d}.png")

    pipe = None
    torch.cuda.empty_cache()


def compute_metrics(model_name):
    gen_path = str(Path(OUTPUT_DIR) / model_name / "generated")

    metrics = calculate_metrics(
        input1=HORSE_DATASET,
        input2=gen_path,
        cuda=torch.cuda.is_available(),
        fid=True,
        isc=True,
        verbose=False,
    )

    fid = round(metrics["frechet_inception_distance"], 3)
    isc = round(metrics["inception_score_mean"], 3)

    wandb.log({"model": model_name, "fid": fid, "isc": isc})

    return {"model": model_name, "FID": fid, "IS": isc}


if __name__ == "__main__":
    wandb.init(project="stage1-model-selection")

    results = []
    for model_name, model_id in MODELS.items():
        train_lora(model_name, model_id)
        generate_images(model_name, model_id)
        results.append(compute_metrics(model_name))

    table = wandb.Table(
        columns=["model", "FID", "IS"],
        data=[[r["model"], r["FID"], r["IS"]] for r in results],
    )
    wandb.log({"results": table})

    best = min(results, key=lambda x: x["FID"])
    wandb.summary["best_model"] = best["model"]
    wandb.summary["best_fid"] = best["FID"]

    wandb.finish()
