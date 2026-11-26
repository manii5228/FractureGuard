from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from fractureguard.data.schema import GENERIC_AUDIT_COLUMNS

SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
}


def iter_image_paths(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            yield path


def select_shard(paths: list[Path], shard_index: int, shard_count: int) -> list[Path]:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be between 0 and shard_count - 1")
    return [path for index, path in enumerate(paths) if index % shard_count == shard_index]


def exclude_audited_paths(paths: list[Path], root: Path, audit_csv: Path) -> list[Path]:
    frame = pd.read_csv(audit_csv, usecols=["image_path"])
    audited = set(frame["image_path"].astype(str))
    return [path for path in paths if path.relative_to(root).as_posix() not in audited]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def infer_bit_depth(image: Image.Image) -> int | None:
    mode_to_bits = {
        "1": 1,
        "L": 8,
        "P": 8,
        "I;16": 16,
        "I;16B": 16,
        "I;16L": 16,
        "I": 32,
        "F": 32,
        "RGB": 8,
        "RGBA": 8,
    }
    return mode_to_bits.get(image.mode)


def inspect_image(path: Path, root: Path, dataset_name: str) -> dict[str, object]:
    relative_path = path.relative_to(root).as_posix()
    row: dict[str, object] = {
        "dataset_name": dataset_name,
        "image_id": path.stem,
        "image_path": relative_path,
        "extension": path.suffix.lower(),
        "width": None,
        "height": None,
        "image_mode": None,
        "source_bit_depth": None,
        "file_size_bytes": path.stat().st_size,
        "file_hash_sha256": sha256_file(path),
        "read_error": None,
    }
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            row.update(
                {
                    "width": image.width,
                    "height": image.height,
                    "image_mode": image.mode,
                    "source_bit_depth": infer_bit_depth(image),
                }
            )
    except Exception as error:  # audit must record bad files instead of stopping
        row["read_error"] = f"{type(error).__name__}: {error}"
    return row


def audit_images(
    root: Path,
    dataset_name: str,
    output_csv: Path,
    paths: Iterable[Path] | None = None,
) -> pd.DataFrame:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    image_paths = list(paths if paths is not None else iter_image_paths(root))
    rows = [
        inspect_image(path, root=root, dataset_name=dataset_name)
        for path in tqdm(
            image_paths,
            desc=f"Auditing {dataset_name}",
            unit="image",
            disable=not sys.stderr.isatty(),
        )
    ]
    frame = pd.DataFrame(rows, columns=GENERIC_AUDIT_COLUMNS)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def merge_audit_files(input_csvs: list[Path], output_csv: Path) -> pd.DataFrame:
    if not input_csvs:
        raise ValueError("At least one audit CSV is required")
    frames = [pd.read_csv(path) for path in input_csvs]
    frame = pd.concat(frames, ignore_index=True)
    duplicates = frame[frame.duplicated(["dataset_name", "image_path"], keep=False)]
    if not duplicates.empty:
        raise ValueError("Audit shards contain duplicate dataset/image_path rows")
    frame = frame.sort_values(["dataset_name", "image_path"], kind="stable")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def summarize_audit(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "images": 0,
            "read_errors": 0,
            "duplicate_files": 0,
            "extensions": {},
            "modes": {},
        }
    duplicate_mask = frame.duplicated("file_hash_sha256", keep=False)
    return {
        "images": int(len(frame)),
        "read_errors": int(frame["read_error"].notna().sum()),
        "duplicate_files": int(duplicate_mask.sum()),
        "extensions": frame["extension"].value_counts(dropna=False).to_dict(),
        "modes": frame["image_mode"].value_counts(dropna=False).to_dict(),
        "min_width": int(frame["width"].dropna().min()) if frame["width"].notna().any() else None,
        "max_width": int(frame["width"].dropna().max()) if frame["width"].notna().any() else None,
        "min_height": int(frame["height"].dropna().min())
        if frame["height"].notna().any()
        else None,
        "max_height": int(frame["height"].dropna().max())
        if frame["height"].notna().any()
        else None,
    }
