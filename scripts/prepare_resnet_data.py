from __future__ import annotations

import random
from pathlib import Path
from shutil import copy2, rmtree


DATA_DIR = Path("data")
RESNET_DATA_DIR = Path("resnet_data")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

DATASETS = ["cifar10", "eurosat_rgb", "beans"]


def copy_dataset_original(dataset_name: str, source_dir: Path, target_dir: Path) -> int:
    """Copy original dataset to original folder."""
    target_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    if source_dir.exists():
        for class_dir in sorted(source_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_target = target_dir / class_dir.name
            class_target.mkdir(exist_ok=True)

            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                copy2(image_path, class_target / image_path.name)
                count += 1

    return count


def setup_generated_folders(source_dir: Path, target_dir: Path) -> int:
    """Create empty class folders in generated directory."""
    target_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for class_dir in sorted(source_dir.iterdir()):
        if class_dir.is_dir():
            class_target = target_dir / class_dir.name
            class_target.mkdir(exist_ok=True)
            count += 1

    return count


def prepare_resnet_data() -> None:

    for dataset_name in DATASETS:
        source_dir = DATA_DIR / dataset_name

        if not source_dir.exists():
            print(f"Skipping {dataset_name}: {source_dir} does not exist")
            continue

        dataset_resnet_dir = RESNET_DATA_DIR / dataset_name

        print(f"\nPreparing {dataset_name}...")

        # 1. Copy original dataset
        original_dir = dataset_resnet_dir / "original"
        if original_dir.exists():
            rmtree(original_dir)
        original_count = copy_dataset_original(dataset_name, source_dir, original_dir)
        print(f"  Original: copied {original_count} images")

        # 2. Copy for augmented dataset (augmentations applied during training)
        augmented_dir = dataset_resnet_dir / "augmented"
        if augmented_dir.exists():
            rmtree(augmented_dir)
        augmented_count = copy_dataset_original(dataset_name, source_dir, augmented_dir)
        print(
            f"  Augmented: copied {augmented_count} images (transforms applied during training)"
        )

        # 3. Setup generated folders (will be populated by generation script)
        generated_dir = dataset_resnet_dir / "generated"
        if generated_dir.exists():
            rmtree(generated_dir)
        class_count = setup_generated_folders(source_dir, generated_dir)
        print(
            f"  Generated: created {class_count} class folders (empty, ready for generation)"
        )


def main() -> None:
    RESNET_DATA_DIR.mkdir(exist_ok=True)
    random.seed(42)
    prepare_resnet_data()
    print(f"\n✅ Resnet data prepared in {RESNET_DATA_DIR}")
    print("   - original/  : Original images (no transforms)")
    print("   - augmented/ : Original images (transforms applied during training)")
    print("   - generated/ : Generated images via LoRA")


if __name__ == "__main__":
    main()
