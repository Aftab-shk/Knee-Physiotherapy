"""
prepare_dataset.py
===================
Data-loading utilities for the Kaggle "Knee Osteoarthritis Dataset with
Severity Grading" dataset (Kellgren-Lawrence grades 0-4).

This module does not train anything by itself. It is imported by both
training.py and inference.py so that every script agrees on image size,
normalization, augmentation, and how KL-grade folders map to class indices.

Expected folder layout (this is how the dataset ships on Kaggle):

    <DATA_DIR>/
        train/
            0/   *.png          <- Grade 0, Normal
            1/   *.png          <- Grade 1, Doubtful
            2/   *.png          <- Grade 2, Mild
            3/   *.png          <- Grade 3, Moderate
            4/   *.png          <- Grade 4, Severe
        val/
            0/ 1/ 2/ 3/ 4/
        test/
            0/ 1/ 2/ 3/ 4/

Run this file directly to sanity-check a downloaded copy of the dataset:

    python prepare_dataset.py --data_dir "/path/to/Knee Osteoarthritis Dataset with Severity Grading"

If val/ is missing (a few dataset re-uploads only ship train/test), a
stratified validation split is carved out of train/ automatically the first
time this runs, then reused (not re-created) on every run after that.
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Constants shared by prepare_dataset.py / training.py / inference.py
# ---------------------------------------------------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RANDOM_SEED = 42
DEFAULT_OUTPUT_DIR = "outputs"
VAL_FRACTION_OF_TRAIN = 0.15  # only used as a fallback if val/ doesn't exist

# Kellgren-Lawrence (KL) grading system used by this dataset
KL_GRADE_NAMES = {
    0: "Grade 0 - Normal",
    1: "Grade 1 - Doubtful",
    2: "Grade 2 - Mild",
    3: "Grade 3 - Moderate",
    4: "Grade 4 - Severe",
}


# ---------------------------------------------------------------------------
# Locating the dataset
# ---------------------------------------------------------------------------
def _find_train_dirs_under(base: Path, max_depth: int = 5) -> List[Path]:
    """
    Search under `base`, up to `max_depth` levels deep, for any directory
    that directly contains a 'train' subfolder.

    This is intentionally depth-searching rather than assuming a fixed
    nesting level: Kaggle's own mount path for attached datasets has changed
    over time (older notebooks saw /kaggle/input/<dataset>/, newer ones see
    the deeper /kaggle/input/datasets/<owner>/<dataset>/), and re-uploads /
    zip layouts add their own variation on top of that. Searching a few
    levels down is more robust than hard-coding a depth that can go stale.
    """
    found: List[Path] = []

    def _walk(d: Path, depth: int) -> None:
        if not d.is_dir():
            return
        if (d / "train").is_dir():
            found.append(d)
            return  # a match - no need to look further inside it
        if depth >= max_depth:
            return
        try:
            children = [c for c in d.iterdir() if c.is_dir()]
        except (PermissionError, OSError):
            return
        for c in sorted(children):
            _walk(c, depth + 1)

    _walk(base, 0)
    return found


def find_dataset_root(data_dir: Optional[str] = None) -> Path:
    """
    Resolve the dataset root, i.e. the folder that directly contains train/.

    Search order:
      1. `data_dir`, if given (also searched a few levels down, in case it
         points at a parent folder rather than the exact dataset folder)
      2. Kaggle notebook input mounts, searched recursively under /kaggle/input
      3. the current working directory
    """
    candidates: List[Path] = []
    if data_dir:
        given = Path(data_dir).expanduser()
        candidates.append(given)
        candidates.extend(_find_train_dirs_under(given))

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(_find_train_dirs_under(kaggle_input))

    candidates.append(Path.cwd())

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_dir() and (candidate / "train").is_dir():
            return candidate.resolve()

    searched = "\n  - ".join(str(c) for c in candidates) or "(nothing to search)"
    raise FileNotFoundError(
        "Could not find the dataset. Looked for a folder containing a "
        f"'train' subfolder in:\n  - {searched}\n\n"
        "Fix this by passing the correct path explicitly, e.g.\n"
        '  python training.py --data_dir "/path/to/Knee Osteoarthritis Dataset with Severity Grading"'
    )


# ---------------------------------------------------------------------------
# Verifying / repairing the train-val-test split
# ---------------------------------------------------------------------------
def _list_class_dirs(split_dir: Path) -> List[Path]:
    if not split_dir.is_dir():
        return []
    return sorted(d for d in split_dir.iterdir() if d.is_dir())


def _create_val_split_from_train(root: Path, val_fraction: float, seed: int) -> None:
    """Move a stratified slice of train/ into val/, one time only, in place."""
    train_dir = root / "train"
    val_dir = root / "val"
    rng = random.Random(seed)

    print(
        f"No usable val/ folder found under {root} - creating one by moving "
        f"{val_fraction:.0%} of each class out of train/ (done once; safe to re-run)."
    )

    for class_dir in _list_class_dirs(train_dir):
        target_dir = val_dir / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(f for f in class_dir.iterdir() if f.is_file())
        rng.shuffle(files)
        n_val = max(1, round(len(files) * val_fraction)) if files else 0

        for f in files[:n_val]:
            shutil.move(str(f), str(target_dir / f.name))

        print(f"  class '{class_dir.name}': moved {n_val}/{len(files)} images to val/")


def verify_dataset(root: Path) -> Dict[str, bool]:
    """
    Make sure train/ exists (required) and val/ exists (auto-created from
    train/ if missing). Returns which splits are present, e.g.
    {'train': True, 'val': True, 'test': False}.
    """
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Expected a 'train' folder inside {root}, found none.")

    train_classes = _list_class_dirs(train_dir)
    if not train_classes:
        raise FileNotFoundError(f"No class subfolders (e.g. 0,1,2,3,4) found inside {train_dir}.")

    if not _list_class_dirs(val_dir):
        _create_val_split_from_train(root, VAL_FRACTION_OF_TRAIN, RANDOM_SEED)

    present = {
        "train": bool(_list_class_dirs(train_dir)),
        "val": bool(_list_class_dirs(val_dir)),
        "test": bool(_list_class_dirs(test_dir)),
    }

    if not present["test"]:
        print(
            f"Note: no 'test' folder found under {root}. Training will still work, "
            "but there won't be a held-out final evaluation after training."
        )

    return present


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_train_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    """
    Augmentation is deliberately mild and geometry-preserving. The images in
    this dataset are already tightly cropped around the knee joint, so an
    aggressive random-crop risks cutting out the joint space itself - which
    is exactly the feature the KL grade depends on. Horizontal flip is safe
    (it just mirrors left/right knees) and matches the augmentation used in
    the published work on this dataset.
    """
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Deterministic preprocessing used for validation, test, and inference."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Labels and class imbalance
# ---------------------------------------------------------------------------
def build_label_maps(classes: List[str]) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    `classes` is what torchvision.datasets.ImageFolder returns: folder names
    ('0'..'4') sorted alphabetically, in the same order as the model's
    output indices. Returns:
      idx_to_grade: {0: 0, 1: 1, ...}              model index -> KL grade
      idx_to_name : {0: 'Grade 0 - Normal', ...}    model index -> readable label
    """
    idx_to_grade = {i: int(name) for i, name in enumerate(classes)}
    idx_to_name = {
        i: KL_GRADE_NAMES.get(grade, f"Grade {grade}") for i, grade in idx_to_grade.items()
    }
    return idx_to_grade, idx_to_name


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """Inverse-frequency class weights (mean-normalized to 1.0) for CrossEntropyLoss."""
    counts = [0] * len(dataset.classes)
    for _, label in dataset.samples:
        counts[label] += 1
    counts_t = torch.tensor(counts, dtype=torch.float)
    weights = 1.0 / torch.clamp(counts_t, min=1)
    weights = weights * (len(weights) / weights.sum())
    return weights


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
def get_dataloaders(
    data_dir: Optional[str] = None,
    batch_size: int = 32,
    img_size: int = IMG_SIZE,
    num_workers: int = 2,
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], List[str], torch.Tensor]:
    """Returns (train_loader, val_loader, test_loader_or_None, classes, class_weights)."""
    root = find_dataset_root(data_dir)
    present = verify_dataset(root)

    train_ds = datasets.ImageFolder(root / "train", transform=get_train_transform(img_size))
    val_ds = datasets.ImageFolder(root / "val", transform=get_eval_transform(img_size))
    test_ds = (
        datasets.ImageFolder(root / "test", transform=get_eval_transform(img_size))
        if present["test"]
        else None
    )

    if val_ds.class_to_idx != train_ds.class_to_idx:
        raise RuntimeError(
            "train/ and val/ don't have matching class subfolders "
            f"({train_ds.class_to_idx} vs {val_ds.class_to_idx}). Check the dataset "
            "folder for missing or extra class subfolders."
        )
    if test_ds is not None and test_ds.class_to_idx != train_ds.class_to_idx:
        raise RuntimeError(
            "train/ and test/ don't have matching class subfolders "
            f"({train_ds.class_to_idx} vs {test_ds.class_to_idx})."
        )

    class_weights = compute_class_weights(train_ds)

    sampler = None
    shuffle = True
    if use_weighted_sampler:
        sample_weights = [class_weights[label].item() for _, label in train_ds.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = (
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        if test_ds is not None
        else None
    )

    return train_loader, val_loader, test_loader, train_ds.classes, class_weights


# ---------------------------------------------------------------------------
# Standalone sanity check: `python prepare_dataset.py --data_dir ...`
# ---------------------------------------------------------------------------
def _print_split_summary(name: str, ds: datasets.ImageFolder) -> Dict[str, int]:
    counts = [0] * len(ds.classes)
    for _, label in ds.samples:
        counts[label] += 1
    total = sum(counts)
    print(f"\n{name} - {total} images")
    for cls_name, count in zip(ds.classes, counts):
        grade = int(cls_name)
        pct = 100 * count / total if total else 0
        print(f"  {KL_GRADE_NAMES.get(grade, cls_name):22s}: {count:5d} images ({pct:5.1f}%)")
    return dict(zip(ds.classes, counts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify / prepare the Knee OA severity dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to the dataset root (the folder containing train/val/test). "
        "If omitted, common Kaggle input paths are searched automatically.",
    )
    args = parser.parse_args()

    root = find_dataset_root(args.data_dir)
    print(f"Dataset root: {root}")
    present = verify_dataset(root)

    summary = {}
    for split in ("train", "val", "test"):
        if not present[split]:
            continue
        ds = datasets.ImageFolder(root / split, transform=get_eval_transform())
        summary[split] = _print_split_summary(split, ds)

    out_dir = Path(DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved a class-distribution summary to {out_dir / 'dataset_summary.json'}")
    print(f'Dataset looks good. Next: python training.py --data_dir "{root}"')


if __name__ == "__main__":
    main()