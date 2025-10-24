"""Utility to split the cardiac MRI dataset into SynFoC-3D index files.

The training script expects three text files listing image/label paths for the
labelled and validation subsets plus another list with only image paths for the
unlabelled pool.  This helper scans a directory that follows the structure
described by the user (each patient has ``*_frameXX.nii`` and matching
``*_frameXX_gt.nii`` files) and writes ``labeled.txt``, ``unlabeled.txt``,
``val.txt`` and ``test.txt`` with a deterministic train/val/test split.

Example
-------
```
python -m SynFoC-3D.code.utils.prepare_dataset \
    --dataset-root /data/training \
    --output-dir /data/SynFoC3D_indices \
    --train-count 70 --val-count 10 --test-count 20 \
    --relative-paths
```
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class Sample:
    """Container with the image and label path of one cardiac frame."""

    image: Path
    label: Path


def _collect_samples(patient_dir: Path) -> List[Sample]:
    """Return all frame samples found in ``patient_dir``.

    The dataset provides two frames per patient (end-diastolic and
    end-systolic).  We match ``*_frameXX.nii`` images with their corresponding
    ``*_frameXX_gt.nii`` label volumes and ignore non-frame files such as the
    4D scans or metadata.
    """

    samples: List[Sample] = []
    prefix = patient_dir.name
    for img_path in sorted(patient_dir.glob(f"{prefix}_frame*.nii")):
        if img_path.name.endswith("_gt.nii"):
            continue
        label_path = img_path.with_name(img_path.stem + "_gt.nii")
        if label_path.exists():
            samples.append(Sample(image=img_path, label=label_path))
    return samples


def _flatten(patients: Iterable[Sequence[Sample]]) -> List[Sample]:
    out: List[Sample] = []
    for group in patients:
        out.extend(group)
    return out


def _to_output_path(path: Path, base: Path, make_relative: bool) -> Path:
    if not make_relative:
        return path
    try:
        return path.relative_to(base)
    except ValueError:
        return path


def _write_pairs(path: Path, samples: Sequence[Sample], make_relative: bool, base: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            img = _to_output_path(sample.image, base, make_relative)
            lab = _to_output_path(sample.label, base, make_relative)
            f.write(f"{img},{lab}\n")


def _write_images(path: Path, samples: Sequence[Sample], make_relative: bool, base: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            img = _to_output_path(sample.image, base, make_relative)
            f.write(f"{img}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to the ACDC-style dataset root (contains patient folders).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where the split text files will be written.")
    parser.add_argument("--train-count", type=int, default=70, help="Number of patients used for the labelled training split (default: 70).")
    parser.add_argument("--val-count", type=int, default=10, help="Number of patients used for validation (default: 10).")
    parser.add_argument("--test-count", type=int, default=20, help="Number of patients used for testing (default: 20).")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for shuffling patients before splitting (default: 2024).")
    parser.add_argument("--relative-paths", action="store_true", help="Emit paths relative to --dataset-root instead of absolute paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset_root.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_dirs = [p for p in dataset_root.iterdir() if p.is_dir() and p.name.lower().startswith("patient")]
    if not patient_dirs:
        raise FileNotFoundError(f"No patient folders found under {dataset_root}")

    patient_dirs.sort()
    random.Random(args.seed).shuffle(patient_dirs)

    total_required = args.train_count + args.val_count + args.test_count
    if total_required > len(patient_dirs):
        raise ValueError(
            f"Requested {total_required} patients but only {len(patient_dirs)} are available."
        )

    train_dirs = patient_dirs[: args.train_count]
    val_dirs = patient_dirs[args.train_count : args.train_count + args.val_count]
    test_dirs = patient_dirs[args.train_count + args.val_count : total_required]

    train_samples = _flatten(_collect_samples(d) for d in train_dirs)
    val_samples = _flatten(_collect_samples(d) for d in val_dirs)
    test_samples = _flatten(_collect_samples(d) for d in test_dirs)

    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError("One of the splits ended up empty; please check the dataset structure.")

    _write_pairs(output_dir / "labeled.txt", train_samples, args.relative_paths, dataset_root)
    _write_pairs(output_dir / "val.txt", val_samples, args.relative_paths, dataset_root)
    _write_pairs(output_dir / "test.txt", test_samples, args.relative_paths, dataset_root)
    _write_images(output_dir / "unlabeled.txt", train_samples, args.relative_paths, dataset_root)

    summary = (
        f"Created dataset splits under {output_dir} using {len(train_dirs)} train, "
        f"{len(val_dirs)} val and {len(test_dirs)} test patients."
    )
    print(summary)


if __name__ == "__main__":
    main()