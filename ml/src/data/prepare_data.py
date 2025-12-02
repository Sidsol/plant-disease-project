"""
Prepare PlantVillage dataset for PyTorch ImageFolder.

Splits raw image data into train/val/test directories with a configurable ratio.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Split image dataset into train/val/test for PyTorch ImageFolder."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source directory containing class subfolders with images.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Destination directory for the split dataset.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1).",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of data for testing (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite destination directory if it exists.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Validate that split ratios sum to 1.0."""
    total = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total <= 1.01):  # Allow small floating point tolerance
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio} = {total}"
        )
    if any(r < 0 for r in [train_ratio, val_ratio, test_ratio]):
        raise ValueError("All ratios must be non-negative.")


def split_files(
    files: list[Path], train_ratio: float, val_ratio: float
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split a list of files into train, val, test sets."""
    n = len(files)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    return train_files, val_files, test_files


def copy_files_with_progress(
    files: list[Path], dest_dir: Path, split_name: str, class_name: str
) -> int:
    """Copy files to destination with progress updates every 10 files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for i, src_file in enumerate(files, 1):
        dst_file = dest_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        copied += 1

        if i % 10 == 0 or i == len(files):
            print(f"  [{split_name}] {class_name}: {i}/{len(files)} files copied")

    return copied


def prepare_dataset(
    source: Path,
    dest: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    force: bool,
) -> None:
    """Main function to prepare the dataset split."""
    # Validate inputs
    validate_ratios(train_ratio, val_ratio, test_ratio)

    if not source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source}")

    if dest.exists():
        if force:
            print(f"Removing existing destination directory: {dest}")
            shutil.rmtree(dest)
        else:
            raise FileExistsError(
                f"Destination directory already exists: {dest}. Use --force to overwrite."
            )

    # Set random seed for reproducibility
    random.seed(seed)

    # Get all class directories
    class_dirs = [d for d in source.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in: {source}")

    print(f"Found {len(class_dirs)} classes in {source}")
    print(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print(f"Random seed: {seed}")
    print("-" * 60)

    # Statistics
    total_stats = {"train": 0, "val": 0, "test": 0}

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name

        # Collect all image files (common formats)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
        files = [
            f
            for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not files:
            print(f"Warning: No images found in {class_name}, skipping.")
            continue

        # Shuffle and split
        random.shuffle(files)
        train_files, val_files, test_files = split_files(files, train_ratio, val_ratio)

        print(f"\nClass: {class_name}")
        print(f"  Total images: {len(files)}")
        print(f"  Split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

        # Copy files to respective directories
        splits = [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ]

        for split_name, split_files_list in splits:
            if split_files_list:
                split_dest = dest / split_name / class_name
                copied = copy_files_with_progress(
                    split_files_list, split_dest, split_name, class_name
                )
                total_stats[split_name] += copied

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Destination: {dest}")
    print("Total files copied:")
    print(f"  train: {total_stats['train']}")
    print(f"  val:   {total_stats['val']}")
    print(f"  test:  {total_stats['test']}")
    print(f"  TOTAL: {sum(total_stats.values())}")


def main() -> None:
    """Entry point for the script."""
    args = parse_args()

    try:
        prepare_dataset(
            source=args.source,
            dest=args.dest,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            force=args.force,
        )
        print("\nDataset preparation complete!")
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
