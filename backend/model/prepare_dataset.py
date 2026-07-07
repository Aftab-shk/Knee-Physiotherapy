"""
prepare_dataset.py
==================
Splits a flat KL-grade dataset into train / val / test folders
ready for train.py.

Handles all common folder naming conventions automatically:
  0, 1, 2, 3, 4
  Normal, Doubtful, Minimal, Moderate, Severe
  grade_0, grade_1, ...
  KL0, KL1, ...

Output:
  data/
    train/  0/  1/  2/  3/  4/   ← 80% of each grade
    val/    0/  1/  2/  3/  4/   ← 10%
    test/   0/  1/  2/  3/  4/   ← 10%

Usage:
  # Basic
  python model/prepare_dataset.py --src combined --dst data

  # Overwrite an existing data/ folder from a previous run
  python model/prepare_dataset.py --src combined --dst data --overwrite

  # Custom split ratios (train/val/test must sum to 1.0)
  python model/prepare_dataset.py --src combined --dst data --val 0.15 --test 0.15

  # Verify an existing data/ folder without copying anything
  python model/prepare_dataset.py --src combined --dst data --verify
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Grade name mapping — covers all common dataset naming conventions
# ---------------------------------------------------------------------------

NAME_TO_GRADE = {
    # Numeric
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    # grade_N
    "grade_0": 0, "grade_1": 1, "grade_2": 2, "grade_3": 3, "grade_4": 4,
    "grade0":  0, "grade1":  1, "grade2":  2, "grade3":  3, "grade4":  4,
    # KLN
    "kl0": 0, "kl1": 1, "kl2": 2, "kl3": 3, "kl4": 4,
    "kl_0": 0, "kl_1": 1, "kl_2": 2, "kl_3": 3, "kl_4": 4,
    # Named (OSAIL, Mendeley)
    "normal": 0, "doubtful": 1, "minimal": 2, "moderate": 3, "severe": 4,
    "healthy": 0,
    "osteoarthritis_doubtful": 1,
    "osteoarthritis_minimal": 2,
    "osteoarthritis_moderate": 3,
    "osteoarthritis_severe": 4,
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NUM_CLASSES = 5
SEED        = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_grade_folders(root: Path) -> dict:
    """
    Returns {grade_int: Path} for grade folders found directly under root.
    Searches two levels deep to handle wrapper directories.
    """
    found = {}
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.lower() in NAME_TO_GRADE:
            found[NAME_TO_GRADE[d.name.lower()]] = d
    return found


def search_nested(root: Path) -> tuple[dict, Path]:
    """
    Searches up to 2 levels deep for grade folders.
    Returns (grade_folders, folder_they_were_found_in).
    """
    # Level 0 — root itself
    folders = find_grade_folders(root)
    if folders:
        return folders, root

    # Level 1 — immediate children
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            folders = find_grade_folders(sub)
            if folders:
                return folders, sub

    # Level 2 — grandchildren
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            for subsub in sorted(sub.iterdir()):
                if subsub.is_dir():
                    folders = find_grade_folders(subsub)
                    if folders:
                        return folders, subsub

    return {}, root


def collect_images(folder: Path) -> list:
    """Collect all image paths under a folder (non-recursive)."""
    imgs = []
    for ext in IMAGE_EXTS:
        imgs.extend(folder.glob(f"*{ext}"))
        imgs.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(imgs))


def is_valid_image(path: Path) -> bool:
    """Quick sanity check — file must be readable and > 1 KB."""
    try:
        return path.stat().st_size > 1024
    except OSError:
        return False


def make_unique_name(dest_dir: Path, original_name: str) -> str:
    """
    If a file with original_name already exists in dest_dir, append a counter
    to the stem so we never silently overwrite a file from another dataset.
    e.g. image.jpg → image_1.jpg → image_2.jpg
    """
    candidate = Path(original_name)
    dest      = dest_dir / original_name
    if not dest.exists():
        return original_name
    counter = 1
    while True:
        new_name = f"{candidate.stem}_{counter}{candidate.suffix}"
        if not (dest_dir / new_name).exists():
            return new_name
        counter += 1


def split_files(files: list, val_ratio: float, test_ratio: float, seed: int) -> dict:
    """Shuffle and split files into train / val / test."""
    random.seed(seed)
    shuffled  = files[:]
    random.shuffle(shuffled)
    n         = len(shuffled)
    n_val     = max(1, int(n * val_ratio))
    n_test    = max(1, int(n * test_ratio))
    n_train   = n - n_val - n_test
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train : n_train + n_val],
        "test":  shuffled[n_train + n_val :],
    }


def print_bar(label: str, count: int, total: int, width: int = 30) -> None:
    """Print a compact ASCII progress bar."""
    filled = int(width * count / max(total, 1))
    bar    = "█" * filled + "░" * (width - filled)
    pct    = 100 * count / max(total, 1)
    print(f"  {label}  [{bar}]  {count:>5} / {total}  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def verify_only(dst: Path) -> None:
    """Print counts of an existing data/ folder without copying anything."""
    print(f"\nVerifying: {dst.resolve()}\n")
    splits = ["train", "val", "test"]
    header = f"  {'Grade':<8}" + "".join(f"{s:>8}" for s in splits) + f"{'Total':>8}"
    print(header)
    print(f"  {'-' * (8 + 8 * len(splits) + 8)}")
    totals = {s: 0 for s in splits}
    for grade in range(NUM_CLASSES):
        counts = {}
        for split in splits:
            d = dst / split / str(grade)
            counts[split] = len(list(d.glob("*"))) if d.exists() else 0
            totals[split] += counts[split]
        row_total = sum(counts.values())
        print(f"  KL {grade:<5}" + "".join(f"{counts[s]:>8}" for s in splits) + f"{row_total:>8}")
    print(f"  {'-' * (8 + 8 * len(splits) + 8)}")
    grand = sum(totals.values())
    print(f"  {'Total':<8}" + "".join(f"{totals[s]:>8}" for s in splits) + f"{grand:>8}")
    print()


def prepare(src: Path, dst: Path, val_ratio: float, test_ratio: float,
            seed: int, overwrite: bool) -> None:

    # ── Handle existing destination ─────────────────────────────────────────
    if dst.exists():
        if overwrite:
            print(f"⚠️  Removing existing output folder: {dst.resolve()}")
            shutil.rmtree(dst)
        else:
            existing_count = sum(1 for _ in dst.rglob("*") if _.is_file())
            if existing_count > 0:
                print(
                    f"\n⚠️  Destination '{dst}' already contains {existing_count} files.\n"
                    f"   To start fresh, re-run with --overwrite.\n"
                    f"   Continuing will add new files on top of existing ones.\n"
                )

    # ── Find grade folders ──────────────────────────────────────────────────
    grade_folders, found_in = search_nested(src)

    if not grade_folders:
        print("\n❌ Could not detect grade folders. Folders found:")
        for d in sorted(src.rglob("*")):
            if d.is_dir():
                print(f"   {d}")
        print("\nExpected names: 0-4, Normal/Doubtful/Minimal/Moderate/Severe, grade_0-4, KL0-4")
        sys.exit(1)

    if found_in != src:
        print(f"Found grade folders inside sub-folder: {found_in.relative_to(src)}/\n")

    # ── Show what was detected ──────────────────────────────────────────────
    print("Detected grade folders:")
    all_images = {}
    skipped    = 0
    for grade in range(NUM_CLASSES):
        if grade not in grade_folders:
            print(f"  KL {grade}  →  ⚠️  NOT FOUND (will be empty in output)")
            all_images[grade] = []
            continue
        folder = grade_folders[grade]
        imgs   = collect_images(folder)
        valid  = [f for f in imgs if is_valid_image(f)]
        bad    = len(imgs) - len(valid)
        if bad:
            print(f"  KL {grade}  →  {folder.name}/  ({len(valid)} valid, {bad} skipped — too small/corrupt)")
            skipped += bad
        else:
            print(f"  KL {grade}  →  {folder.name}/  ({len(valid)} images)")
        all_images[grade] = valid

    total_images = sum(len(v) for v in all_images.values())
    print(f"\nTotal valid images: {total_images}")
    if skipped:
        print(f"Skipped (corrupt):  {skipped}")
    if total_images == 0:
        print("\n❌ No valid images found. Check your --src path.")
        sys.exit(1)

    print(f"\nSplit: train {(1-val_ratio-test_ratio)*100:.0f}% / "
          f"val {val_ratio*100:.0f}% / test {test_ratio*100:.0f}%\n")

    # ── Copy files ──────────────────────────────────────────────────────────
    summary      = {}
    total_copied = 0

    for grade in range(NUM_CLASSES):
        images = all_images[grade]
        if not images:
            summary[grade] = {"train": 0, "val": 0, "test": 0}
            continue

        splits  = split_files(images, val_ratio, test_ratio, seed)
        summary[grade] = {s: len(v) for s, v in splits.items()}

        for split_name, file_list in splits.items():
            out_dir = dst / split_name / str(grade)
            out_dir.mkdir(parents=True, exist_ok=True)
            for fp in file_list:
                unique_name = make_unique_name(out_dir, fp.name)
                shutil.copy2(fp, out_dir / unique_name)
                total_copied += 1

        # Progress bar per grade
        grade_total = len(images)
        print_bar(f"KL {grade}", grade_total, total_images)

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n  {'Grade':<8} {'Train':>8} {'Val':>6} {'Test':>6} {'Total':>8}")
    print(f"  {'-'*40}")
    totals = {"train": 0, "val": 0, "test": 0}
    for g in range(NUM_CLASSES):
        s     = summary[g]
        total = s["train"] + s["val"] + s["test"]
        print(f"  KL {g:<5} {s['train']:>8} {s['val']:>6} {s['test']:>6} {total:>8}")
        for k in totals:
            totals[k] += s[k]
    print(f"  {'-'*40}")
    grand = sum(totals.values())
    print(f"  {'Total':<8} {totals['train']:>8} {totals['val']:>6} {totals['test']:>6} {grand:>8}")

    print(f"\n✅ Dataset ready at: {dst.resolve()}")
    print(f"\nNext step:")
    print(f"  python train.py --data {args_dst_str} --epochs 40 --batch-size 16")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

args_dst_str = "data"   # updated after args parse — used in print


def main():
    global args_dst_str

    parser = argparse.ArgumentParser(
        description="Prepare knee X-ray dataset for KL-grade classifier training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--src",       required=True,
                        help="Root folder containing grade sub-folders")
    parser.add_argument("--dst",       default="data",
                        help="Output directory (default: data/)")
    parser.add_argument("--val",       type=float, default=0.10,
                        help="Validation split ratio (default: 0.10)")
    parser.add_argument("--test",      type=float, default=0.10,
                        help="Test split ratio (default: 0.10)")
    parser.add_argument("--seed",      type=int,   default=SEED,
                        help="Random seed for reproducible splits (default: 42)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete and recreate destination folder before splitting")
    parser.add_argument("--verify",    action="store_true",
                        help="Print counts of an existing output folder without copying")
    args = parser.parse_args()

    # Validate split ratios
    if not (0.0 < args.val < 1.0 and 0.0 < args.test < 1.0):
        print("❌ --val and --test must each be between 0.0 and 1.0")
        sys.exit(1)
    if args.val + args.test >= 1.0:
        print("❌ --val + --test must be less than 1.0 (leaves something for training)")
        sys.exit(1)

    src          = Path(args.src)
    dst          = Path(args.dst)
    args_dst_str = args.dst

    if not src.exists():
        print(f"❌ Source folder not found: {src.resolve()}")
        sys.exit(1)

    print(f"\nSource : {src.resolve()}")
    print(f"Output : {dst.resolve()}")

    if args.verify:
        verify_only(dst)
        return

    prepare(src, dst, args.val, args.test, args.seed, args.overwrite)


if __name__ == "__main__":
    main()