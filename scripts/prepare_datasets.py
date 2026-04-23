import os
import numpy as np
from pathlib import Path
from PIL import Image

import hydra
from omegaconf import DictConfig

from torch.utils.data import Dataset, Subset
from torchvision import datasets as tvdatasets
from torchvision import transforms


CLASS_TEMPLATES = {
    "cifar10": [
        "a photo of an airplane",
        "a photo of a car",
        "a photo of a bird",
        "a photo of a cat",
        "a photo of a deer",
        "a photo of a dog",
        "a photo of a frog",
        "a photo of a horse",
        "a photo of a ship",
        "a photo of a truck",
    ],
    "flowers102": [f"a photo of a flower species {i}" for i in range(102)],
    "oxford_pets": [
        "a photo of an abyssinian cat",
        "a photo of an american bulldog",
        "a photo of an american pit bull terrier",
        "a photo of a basset hound",
        "a photo of a beagle",
        "a photo of a bengal cat",
        "a photo of a birman cat",
        "a photo of a bombay cat",
        "a photo of a boxer dog",
        "a photo of a british shorthair cat",
        "a photo of a chihuahua",
        "a photo of an english cocker spaniel",
        "a photo of an english setter",
        "a photo of a german shorthaired pointer",
        "a photo of a great pyrenees",
        "a photo of a havanese dog",
        "a photo of a japanese chin",
        "a photo of a keeshond",
        "a photo of a leonberger",
        "a photo of a maine coon cat",
        "a photo of a miniature pinscher",
        "a photo of a newfoundland dog",
        "a photo of a persian cat",
        "a photo of a pomeranian",
        "a photo of a pug",
        "a photo of a ragdoll cat",
        "a photo of a russian blue cat",
        "a photo of a saint bernard",
        "a photo of a samoyed",
        "a photo of a scottish terrier",
        "a photo of a shiba inu",
        "a photo of a siamese cat",
        "a photo of a sphynx cat",
        "a photo of a staffordshire bull terrier",
        "a photo of a wheaten terrier",
        "a photo of a yorkshire terrier",
        "a photo of an egyptian mau cat",
    ],
    "stanford_cars": [f"a photo of a {1990 + i} car model" for i in range(196)],
    "eurosat": [
        "a satellite image of annual crop land",
        "a satellite image of forest",
        "a satellite image of herbaceous vegetation",
        "a satellite image of a highway road",
        "a satellite image of industrial buildings",
        "a satellite image of pasture land",
        "a satellite image of permanent crop land",
        "a satellite image of residential buildings",
        "a satellite image of a river",
        "a satellite image of a sea or lake",
    ],
}


class TemplateDataset(Dataset):
    def __init__(self, base_dataset, class_templates, transform=None):
        self.base = base_dataset
        self.templates = class_templates
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        if self.transform and isinstance(image, Image.Image):
            image = self.transform(image)
        prompt = self.templates[label]
        return image, label, prompt


def get_targets(dataset):
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "_labels"):
        return np.array(dataset._labels)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    return np.array([dataset[i][1] for i in range(len(dataset))])


def get_subset(dataset, shots_per_class):
    targets = get_targets(dataset)
    indices = []
    for cls in np.unique(targets):
        cls_idx = np.where(targets == cls)[0]
        indices.extend(cls_idx[:shots_per_class].tolist())
    return Subset(dataset, indices)


def save_to_disk(dataset, out_dir):
    out_dir = Path(out_dir)
    for idx in range(len(dataset)):
        image, label, prompt = dataset[idx]
        class_dir = out_dir / str(label)
        class_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(image, Image.Image):
            image.save(class_dir / f"{idx:05d}.png")
        (class_dir / f"{idx:05d}.txt").write_text(prompt)


@hydra.main(config_path="../conf", config_name="prepare_datasets", version_base=None)
def main(cfg: DictConfig):
    out_root = Path(cfg.output_dir)
    shots = cfg.shots_per_class
    transform = transforms.Compose(
        [
            transforms.Resize((cfg.resolution, cfg.resolution)),
            transforms.ToTensor(),
        ]
    )

    loaders = {
        "cifar10": lambda: tvdatasets.CIFAR10(
            root=cfg.data_root, train=True, download=True
        ),
        "flowers102": lambda: tvdatasets.Flowers102(
            root=cfg.data_root, split="train", download=True
        ),
        "oxford_pets": lambda: tvdatasets.OxfordIIITPet(
            root=cfg.data_root, split="trainval", download=True
        ),
        "stanford_cars": lambda: tvdatasets.StanfordCars(
            root=cfg.data_root, split="train", download=True
        ),
        "eurosat": lambda: tvdatasets.EuroSAT(root=cfg.data_root, download=True),
    }

    active = list(cfg.datasets) if cfg.get("datasets") else list(loaders.keys())

    for name in active:
        print(f"\nLoading {name}...")
        base = loaders[name]()
        subset = get_subset(base, shots)
        dataset = TemplateDataset(subset, CLASS_TEMPLATES[name], transform=transform)

        out_path = out_root / name
        print(f"Saving {len(dataset)} samples to {out_path}...")
        save_to_disk(dataset, out_path)
        print(f"Done — classes: {len(CLASS_TEMPLATES[name])} | shots/class: {shots}")
        print(f"Sample prompt: '{CLASS_TEMPLATES[name][0]}'")

    print(f"\nAll datasets saved to {out_root}")


if __name__ == "__main__":
    main()
