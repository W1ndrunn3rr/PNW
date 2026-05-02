from __future__ import annotations

from pathlib import Path


TRAIN_SCRIPT = Path("diffusers/examples/dreambooth/train_dreambooth_lora_sana.py")

LOAD_DATASET_BLOCK = """            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
"""

LOCAL_METADATA_BLOCK = """            dataset_path = Path(args.dataset_name)
            if dataset_path.exists() and (dataset_path / "metadata.jsonl").exists():
                from datasets import Dataset, DatasetDict, Image

                rows = []
                with (dataset_path / "metadata.jsonl").open("r", encoding="utf-8") as metadata_file:
                    for line in metadata_file:
                        item = json.loads(line)
                        rows.append(
                            {
                                "image": str(dataset_path / item["file_name"]),
                                "text": item["text"],
                            }
                        )
                dataset = DatasetDict({"train": Dataset.from_list(rows).cast_column("image", Image())})
            else:
                dataset = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    cache_dir=args.cache_dir,
                )
"""


def add_import(source: str, import_line: str, after_line: str) -> str:
    if import_line in source:
        return source
    if after_line not in source:
        raise RuntimeError(f"Could not find import anchor: {after_line}")
    return source.replace(after_line, f"{after_line}{import_line}", 1)


def patch_source(source: str) -> str:
    source = add_import(source, "import json\n", "import argparse\n")
    source = add_import(source, "from pathlib import Path\n", "from contextlib import nullcontext\n")

    if LOCAL_METADATA_BLOCK in source:
        return source
    if LOAD_DATASET_BLOCK not in source:
        raise RuntimeError(
            "Could not find the SANA DreamBooth load_dataset block. "
            "The upstream Diffusers script may have changed."
        )
    return source.replace(LOAD_DATASET_BLOCK, LOCAL_METADATA_BLOCK, 1)


def main() -> None:
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(
            f"{TRAIN_SCRIPT} does not exist. Run `make setup` after cloning local Diffusers."
        )

    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    patched = patch_source(source)
    if patched == source:
        print(f"{TRAIN_SCRIPT} already patched")
        return

    TRAIN_SCRIPT.write_text(patched, encoding="utf-8")
    print(f"Patched {TRAIN_SCRIPT}")


if __name__ == "__main__":
    main()
