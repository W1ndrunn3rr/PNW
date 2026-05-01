#!/usr/bin/env python3
"""Stage 3: Train ResNet-18 on all dataset variants with 5-fold cross-validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm


RESNET_DATA_DIR = Path("resnet_data")
OUTPUT_DIR = Path("outputs/stage3")
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

DATASETS = ["cifar10", "eurosat_rgb", "beans"]
AUGMENTATION_TYPES = ["original", "augmented", "generated"]

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_FOLDS = 5


class ImageFolderDataset(Dataset):
    """Load images from a folder structure."""

    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}

        # Build class index
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir.name] = idx

        # Collect images
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


def get_transforms(dataset_name: str) -> tuple:
    """Get dataset-specific transforms."""
    if dataset_name == "cifar10":
        size = 32
    elif dataset_name == "beans":
        size = 224
    else:  # eurosat_rgb
        size = 64

    train_transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, test_transform


def create_model(num_classes: int) -> nn.Module:
    """Create ResNet-18 model."""
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, accuracy, f1


def train_with_kfold(
    dataset_name: str,
    augmentation_type: str,
    dataset: Dataset,
    dataset_path: Path,
) -> dict:
    """Train model with k-fold cross-validation."""

    train_transform, test_transform = get_transforms(dataset_name)

    # Apply augmentations during training only for "augmented" variant
    if augmentation_type == "augmented":
        dataset.transform = train_transform
    else:
        dataset.transform = test_transform

    num_classes = len(set([label for _, label in dataset]))
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    fold_results = {
        "train_losses": [],
        "test_losses": [],
        "accuracies": [],
        "f1_scores": [],
    }

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        print(f"  Fold {fold + 1}/{NUM_FOLDS}...")

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(
            train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        model = create_model(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_f1 = 0.0
        best_model_path = (
            OUTPUT_DIR / f"{dataset_name}_{augmentation_type}_fold{fold}.pt"
        )

        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            test_loss, accuracy, f1 = evaluate(model, test_loader, criterion, DEVICE)
            scheduler.step()

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), best_model_path)

            if (epoch + 1) % 10 == 0:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                        "accuracy": accuracy,
                        "f1_score": f1,
                        "fold": fold + 1,
                        "epoch": epoch + 1,
                    }
                )

        fold_results["train_losses"].append(train_loss)
        fold_results["test_losses"].append(test_loss)
        fold_results["accuracies"].append(accuracy)
        fold_results["f1_scores"].append(f1)

    return fold_results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    wandb.init(project="stage3-resnet", name="cross-validation")

    results = {}

    for dataset_name in DATASETS:
        print(f"\nProcessing {dataset_name}...")

        dataset_results = {}

        for aug_type in AUGMENTATION_TYPES:
            data_path = RESNET_DATA_DIR / dataset_name / aug_type

            if not data_path.exists():
                print(f"  {aug_type} not found at {data_path}, skipping")
                continue

            print(f"  Training on {aug_type}...")

            dataset = ImageFolderDataset(data_path)
            print(
                f"    Loaded {len(dataset)} images, {len(dataset.class_to_idx)} classes"
            )

            fold_results = train_with_kfold(dataset_name, aug_type, dataset, data_path)

            dataset_results[aug_type] = fold_results

            avg_accuracy = np.mean(fold_results["accuracies"])
            avg_f1 = np.mean(fold_results["f1_scores"])
            std_accuracy = np.std(fold_results["accuracies"])
            std_f1 = np.std(fold_results["f1_scores"])

            print(f"    Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
            print(f"    F1-Score: {avg_f1:.4f} ± {std_f1:.4f}")

            wandb.log(
                {
                    f"{dataset_name}_{aug_type}_avg_accuracy": avg_accuracy,
                    f"{dataset_name}_{aug_type}_std_accuracy": std_accuracy,
                    f"{dataset_name}_{aug_type}_avg_f1": avg_f1,
                    f"{dataset_name}_{aug_type}_std_f1": std_f1,
                }
            )

        results[dataset_name] = dataset_results

    # Save results summary
    summary_path = OUTPUT_DIR / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete! Results saved to {OUTPUT_DIR}")
    wandb.finish()


if __name__ == "__main__":
    main()
