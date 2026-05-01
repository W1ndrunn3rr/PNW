from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm
from scipy import stats


RESNET_DATA_DIR = Path("resnet_data")
MODELS_DIR = Path("outputs/stage3")  # where your .pt files are saved
OUTPUT_DIR = Path("outputs/stage3_eval")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = ["cifar10", "eurosat_rgb", "beans"]
AUGMENTATION_TYPES = ["original", "augmented", "generated"]
NUM_FOLDS = 5
BATCH_SIZE = 32
RANDOM_SEED = 42  # must match the one used during training


class ImageFolderDataset(Dataset):
    """Load images from a folder structure. Identical to training script."""

    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}

        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir.name] = idx

        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            for img_path in sorted(class_dir.glob("*.png")):
                self.images.append(img_path)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from PIL import Image

        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(dataset_name: str):
    """Same transforms as training – test_transform only (no augmentation)."""
    if dataset_name == "cifar10":
        size = 32
    elif dataset_name == "beans":
        size = 224
    else:  # eurosat_rgb
        size = 64

    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform


def create_model(num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet18(
        pretrained=False
    )  # we will load weights, no need pretrained
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return acc, f1


def load_and_evaluate_fold(
    dataset_path: Path,
    model_path: Path,
    fold_indices: tuple,
    transform,
    device,
    num_classes: int,
):
    """Load model for one fold and evaluate on the test subset."""
    train_idx, test_idx = fold_indices

    # Full dataset with correct transform
    full_dataset = ImageFolderDataset(dataset_path, transform=transform)
    test_subset = Subset(full_dataset, test_idx)

    test_loader = DataLoader(
        test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = create_model(num_classes, device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    acc, f1 = evaluate_model(model, test_loader, device)
    return acc, f1


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wandb.init(project="PNW", name="stage3-evaluation-statistical")

    # Will store per‑fold metrics for each dataset and augmentation
    all_results = {}

    for dataset_name in DATASETS:
        print(f"\n{'=' * 60}\nDataset: {dataset_name}\n{'=' * 60}")
        transform = get_transforms(dataset_name)
        dataset_results = {}

        # First, load the full dataset once to get the number of classes and the fold indices
        path_original = RESNET_DATA_DIR / dataset_name / "original"
        if not path_original.exists():
            print(f"  {dataset_name}/original not found, skipping dataset")
            continue

        full_dataset = ImageFolderDataset(
            path_original, transform=transform
        )  # transforms don't matter for indices
        num_classes = len(full_dataset.class_to_idx)

        # Create the same KFold split as during training
        kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        fold_splits = list(kfold.split(full_dataset))  # list of (train_idx, test_idx)

        # For each augmentation type, collect per‑fold accuracies and f1 scores
        per_fold_metrics = {aug: {"acc": [], "f1": []} for aug in AUGMENTATION_TYPES}

        for aug_type in AUGMENTATION_TYPES:
            data_path = RESNET_DATA_DIR / dataset_name / aug_type
            if not data_path.exists():
                print(f"  {aug_type} not found, skipping")
                continue

            print(f"\n  Evaluating {aug_type} models...")
            for fold, (train_idx, test_idx) in enumerate(fold_splits):
                model_path = MODELS_DIR / f"{dataset_name}_{aug_type}_fold{fold}.pt"
                if not model_path.exists():
                    print(
                        f"    Warning: model {model_path} not found, skipping fold {fold}"
                    )
                    continue

                acc, f1 = load_and_evaluate_fold(
                    dataset_path=data_path,
                    model_path=model_path,
                    fold_indices=(train_idx, test_idx),
                    transform=transform,
                    device=DEVICE,
                    num_classes=num_classes,
                )
                per_fold_metrics[aug_type]["acc"].append(acc)
                per_fold_metrics[aug_type]["f1"].append(f1)
                print(f"    Fold {fold + 1}: acc={acc:.4f}, f1={f1:.4f}")

            # Print summary for this augmentation type
            if per_fold_metrics[aug_type]["acc"]:
                avg_acc = np.mean(per_fold_metrics[aug_type]["acc"])
                std_acc = np.std(per_fold_metrics[aug_type]["acc"])
                avg_f1 = np.mean(per_fold_metrics[aug_type]["f1"])
                std_f1 = np.std(per_fold_metrics[aug_type]["f1"])
                print(
                    f"    Summary: acc = {avg_acc:.4f} ± {std_acc:.4f}, f1 = {avg_f1:.4f} ± {std_f1:.4f}"
                )
                wandb.log(
                    {
                        f"{dataset_name}_{aug_type}_avg_acc": avg_acc,
                        f"{dataset_name}_{aug_type}_std_acc": std_acc,
                        f"{dataset_name}_{aug_type}_avg_f1": avg_f1,
                        f"{dataset_name}_{aug_type}_std_f1": std_f1,
                    }
                )

        # --- Statistical tests between augmentation types ---
        print(f"\n  --- Statistical comparisons for {dataset_name} ---")
        variants = [v for v in AUGMENTATION_TYPES if per_fold_metrics[v]["acc"]]
        for i, a in enumerate(variants):
            for b in variants[i + 1 :]:
                acc_a = per_fold_metrics[a]["acc"]
                acc_b = per_fold_metrics[b]["acc"]
                # Paired t-test
                t_stat, p_ttest = stats.ttest_rel(acc_a, acc_b)
                # Wilcoxon signed-rank test (non-parametric)
                w_stat, p_wilcox = stats.wilcoxon(acc_a, acc_b)
                print(f"\n    {a} vs {b}:")
                print(f"      t-test:     t = {t_stat:.4f}, p = {p_ttest:.6f}")
                print(f"      Wilcoxon:   W = {w_stat:.1f}, p = {p_wilcox:.6f}")
                wandb.log(
                    {
                        f"{dataset_name}_{a}_vs_{b}_ttest_p": p_ttest,
                        f"{dataset_name}_{a}_vs_{b}_wilcoxon_p": p_wilcox,
                    }
                )

        dataset_results["per_fold_metrics"] = per_fold_metrics
        all_results[dataset_name] = dataset_results

    # Save results to JSON
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(all_results, default=convert_numpy))
    output_path = OUTPUT_DIR / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
