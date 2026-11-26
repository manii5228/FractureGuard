from __future__ import annotations

import hashlib
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

PREPROCESSING_VARIANTS = (
    "percentile_1_99",
    "percentile_1_99_clahe",
    "full_range",
)


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def create_milestone2_cohort(
    ready_manifest_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(ready_manifest_csv)
    initial = _bool_series(frame["initial_exam"])
    uncertain = _bool_series(frame["diagnosis_uncertain"])
    cast = _bool_series(frame["cast"])
    metal = _bool_series(frame["metal"])
    supported = frame["projection"].isin(["frontal", "lateral"])

    train = frame["split"].eq("train") & initial & ~uncertain & supported
    validation = frame["split"].eq("validation") & initial & ~uncertain & ~cast & ~metal & supported
    selected = frame.loc[train | validation].copy()
    selected["m2_split"] = np.where(train.loc[selected.index], "train", "val")
    selected["preprocessing_candidates"] = ",".join(PREPROCESSING_VARIANTS)
    selected = selected.sort_values(["m2_split", "patient_id", "image_id"], kind="stable")
    if selected.groupby("patient_id")["m2_split"].nunique().gt(1).any():
        raise ValueError("A patient crosses Milestone 2 train/validation partitions")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_csv, index=False)
    return selected


def summarize_milestone2_cohort(frame: pd.DataFrame) -> dict[str, object]:
    positive = _bool_series(frame["fracture_visible"])
    return {
        split: {
            "images": int(mask.sum()),
            "patients": int(frame.loc[mask, "patient_id"].nunique()),
            "positive_images": int((mask & positive).sum()),
            "negative_images": int((mask & ~positive).sum()),
            "fracture_boxes": int(frame.loc[mask, "fracture_box_count"].sum()),
        }
        for split in ("train", "val")
        if (mask := frame["m2_split"].eq(split)).any()
    }


def _normalize_image(source_path: Path, variant: str) -> np.ndarray:
    with Image.open(source_path) as image:
        pixels = np.asarray(image, dtype=np.float32)
    if variant.startswith("percentile_1_99"):
        lower, upper = np.percentile(pixels, (1.0, 99.0))
    elif variant == "full_range":
        lower, upper = float(pixels.min()), float(pixels.max())
    else:
        raise ValueError(f"Unknown preprocessing variant: {variant}")
    if upper <= lower:
        upper = lower + 1.0
    normalized = np.clip((pixels - lower) * (255.0 / (upper - lower)), 0, 255).astype(np.uint8)
    if variant == "percentile_1_99_clahe":
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("CLAHE preprocessing requires OpenCV") from error
        normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(normalized)
    return normalized


def _fracture_only_yolo(source_label_path: Path) -> tuple[str, int]:
    output_lines: list[str] = []
    if source_label_path.is_file():
        for line in source_label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if not values:
                continue
            if len(values) != 5:
                raise ValueError(f"Invalid YOLO label row in {source_label_path}: {line}")
            class_id, *coordinates = values
            if class_id != "3":
                continue
            numbers = [float(value) for value in coordinates]
            if not all(0.0 <= value <= 1.0 for value in numbers):
                raise ValueError(f"Out-of-range YOLO label in {source_label_path}: {line}")
            output_lines.append("0 " + " ".join(f"{value:.8f}" for value in numbers))
    text = "\n".join(output_lines)
    if text:
        text += "\n"
    return text, len(output_lines)


def _validate_yolo_text(text: str, context: str) -> int:
    """Check a fracture-only YOLO label and return its box count."""
    count = 0
    for line in text.splitlines():
        values = line.split()
        if not values:
            continue
        if len(values) != 5 or values[0] != "0":
            raise ValueError(f"Invalid fracture-only YOLO row in {context}: {line}")
        x, y, w, h = (float(value) for value in values[1:])
        if w <= 0 or h <= 0:
            raise ValueError(f"Degenerate YOLO box in {context}: {line}")
        tolerance = 1e-6
        if (
            x - w / 2 < -tolerance
            or x + w / 2 > 1 + tolerance
            or y - h / 2 < -tolerance
            or y + h / 2 > 1 + tolerance
        ):
            raise ValueError(f"YOLO box leaves the image in {context}: {line}")
        count += 1
    return count


def load_frozen_labels(labels_csv: Path) -> dict[str, str]:
    """Load the frozen fracture-only labels shipped with the Milestone 2 cohort.

    Each row stores one image's boxes joined by ``|``. Shipping these labels removes any
    dependence on whichever annotation copy a GPU host happens to provide.
    """
    frame = pd.read_csv(labels_csv, dtype={"yolo_boxes": str}, keep_default_na=False)
    required = {"image_id", "fracture_box_count", "yolo_boxes"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Frozen label table missing columns: {sorted(missing)}")
    if frame["image_id"].duplicated().any():
        raise ValueError("Frozen label table contains duplicate image IDs")
    labels: dict[str, str] = {}
    for record in frame.itertuples(index=False):
        rows = [row.strip() for row in str(record.yolo_boxes).split("|") if row.strip()]
        text = "\n".join(rows) + ("\n" if rows else "")
        count = _validate_yolo_text(text, f"frozen labels for {record.image_id}")
        if count != int(record.fracture_box_count):
            raise ValueError(f"Frozen label count mismatch for {record.image_id}")
        labels[str(record.image_id)] = text
    return labels


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(destination: Path, data: bytes) -> None:
    """Write via a temporary sibling and rename, so an interruption never leaves a
    truncated image or label that a later scan would treat as real data."""
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, destination)


def _encode_png(pixels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


def _resolve_source_image(dataset_root: Path, image_path: str) -> Path:
    """Resolve both the extracted local layout and Kaggle's four-part layout."""
    direct = dataset_root / image_path
    if direct.is_file():
        return direct
    filename = Path(image_path).name
    for candidate in (dataset_root / filename, dataset_root / "images" / filename):
        if candidate.is_file():
            return candidate
    for part in ("images_part1", "images_part2", "images_part3", "images_part4"):
        for candidate in (
            dataset_root / part / filename,
            dataset_root / part / part / filename,
            dataset_root / part / "images" / filename,
        ):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Image not found under {dataset_root}: {image_path}")


def _resolve_source_label(dataset_root: Path, image_id: str) -> Path:
    for candidate in (
        dataset_root / "yolov5" / "labels" / f"{image_id}.txt",
        dataset_root / "folder_structure" / "yolov5" / "labels" / f"{image_id}.txt",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"YOLO label not found under {dataset_root}: {image_id}.txt")


HASH_CHECK_MODES = ("strict", "off")


def _build_case(
    task: tuple[dict[str, object], str, str, str, str | None, str],
) -> dict[str, object]:
    record, dataset_root_text, output_root_text, variant, frozen_label, hash_check = task
    image_id = str(record["image_id"])
    split = str(record["m2_split"])
    result: dict[str, object] = {
        "image_id": image_id,
        "m2_split": split,
        "variant": variant,
        "status": "ok",
        "error": "",
    }
    try:
        dataset_root = Path(dataset_root_text)
        output_root = Path(output_root_text)
        source_image = _resolve_source_image(dataset_root, str(record["image_path"]))
        expected_hash = str(record.get("file_hash_sha256") or "")
        if hash_check == "strict" and expected_hash:
            actual_hash = _sha256_file(source_image)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Source SHA-256 differs from the audited file ({actual_hash[:12]} != "
                    f"{expected_hash[:12]}); the host copy is not the audited image"
                )
        if frozen_label is None:
            label_text, _ = _fracture_only_yolo(_resolve_source_label(dataset_root, image_id))
        else:
            label_text = frozen_label
        box_count = _validate_yolo_text(label_text, image_id)
        expected_boxes = int(record["fracture_box_count"])
        if box_count != expected_boxes:
            raise ValueError(
                f"Fracture box mismatch: labels={box_count}, manifest={expected_boxes}"
            )

        normalized = _normalize_image(source_image, variant)
        expected_shape = (int(record.get("height") or 0), int(record.get("width") or 0))
        if all(expected_shape) and normalized.shape != expected_shape:
            raise ValueError(f"Image shape {normalized.shape} != audited {expected_shape}")

        destination_image = output_root / "images" / split / f"{image_id}.png"
        destination_label = output_root / "labels" / split / f"{image_id}.txt"
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(destination_image, _encode_png(normalized))
        _atomic_write_bytes(destination_label, label_text.encode("utf-8"))
        result.update(
            {
                "box_count": box_count,
                "output_image": destination_image.as_posix(),
                "output_label": destination_label.as_posix(),
            }
        )
    except Exception as error:  # noqa: BLE001 - collected and reported for every case
        result.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
    return result


def build_yolo_variant(
    cohort_csv: Path,
    dataset_root: Path,
    output_root: Path,
    variant: str,
    workers: int = 4,
    limit_per_split: int | None = None,
    labels_csv: Path | None = None,
    hash_check: str = "strict",
) -> pd.DataFrame:
    """Build one fracture-only 8-bit YOLO variant from the frozen cohort.

    Every case is attempted; failures are written to ``build_failures.csv`` and the build
    raises without writing ``dataset.yaml``, so a partial dataset can never be trained on.
    """
    if variant not in PREPROCESSING_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {PREPROCESSING_VARIANTS}")
    if hash_check not in HASH_CHECK_MODES:
        raise ValueError(f"hash_check must be one of {HASH_CHECK_MODES}")
    cohort = pd.read_csv(cohort_csv)
    if limit_per_split is not None:
        cohort = pd.concat(
            [group.head(limit_per_split) for _, group in cohort.groupby("m2_split", sort=True)],
            ignore_index=True,
        )
    frozen = load_frozen_labels(labels_csv) if labels_csv is not None else None
    if frozen is not None:
        missing = set(cohort["image_id"].astype(str)).difference(frozen)
        if missing:
            raise ValueError(f"{len(missing)} cohort images have no frozen label")
    variant_root = output_root / variant
    variant_root.mkdir(parents=True, exist_ok=True)
    stale_yaml = variant_root / "dataset.yaml"
    if stale_yaml.exists():
        stale_yaml.unlink()
    tasks = [
        (
            record,
            str(dataset_root.resolve()),
            str(variant_root.resolve()),
            variant,
            None if frozen is None else frozen[str(record["image_id"])],
            hash_check,
        )
        for record in cohort.to_dict(orient="records")
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(_build_case, tasks, chunksize=16))
    result = pd.DataFrame(rows).sort_values(["m2_split", "image_id"], kind="stable")
    failures = result.loc[result["status"].ne("ok")]
    if not failures.empty:
        report = variant_root / "build_failures.csv"
        failures.to_csv(report, index=False)
        examples = "; ".join(failures["error"].head(3))
        raise RuntimeError(
            f"{len(failures)} of {len(result)} cases failed for {variant}; see {report}. "
            f"Examples: {examples}"
        )
    (variant_root / "build_failures.csv").unlink(missing_ok=True)
    expected = cohort.groupby("m2_split")["fracture_box_count"].sum().astype(int).to_dict()
    actual = result.groupby("m2_split")["box_count"].sum().astype(int).to_dict()
    if actual != expected:
        raise ValueError(
            f"Box totals changed during conversion: expected={expected}, actual={actual}"
        )

    metadata = {
        "variant": variant,
        "source_cohort": str(cohort_csv),
        "source_dataset_root": str(dataset_root),
        "label_source": "frozen_labels_csv" if frozen is not None else "source_yolov5_labels",
        "source_hash_check": hash_check,
        "images": int(len(result)),
        "splits": result["m2_split"].value_counts().to_dict(),
        "fracture_boxes": int(result["box_count"].sum()),
        "parameters": {
            "percentile_window": [1.0, 99.0] if variant.startswith("percentile") else None,
            "clahe_clip_limit": 2.0 if variant.endswith("clahe") else None,
            "clahe_tile_grid": [8, 8] if variant.endswith("clahe") else None,
            "output_bit_depth": 8,
            "geometry_changed": False,
        },
    }
    (variant_root / "build_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    result.drop(columns=["status", "error"]).to_csv(
        variant_root / "build_manifest.csv", index=False
    )
    write_dataset_yaml(variant_root)
    return result


def write_dataset_yaml(variant_root: Path, output: Path | None = None) -> Path:
    """Write a train/val-only Ultralytics dataset YAML pointing at this host's copy.

    ``output`` lets a read-only prebuilt variant (e.g. a Kaggle input) get a YAML elsewhere.
    """
    dataset_yaml = {
        "path": str(variant_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "fracture"},
    }
    output = output or variant_root / "dataset.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_yaml, handle, sort_keys=False)
    return output


def _verify_image(task: tuple[str, int, int]) -> str:
    path_text, height, width = task
    try:
        with Image.open(path_text) as image:
            image.load()
            if image.mode != "L":
                return f"mode {image.mode} is not 8-bit grayscale"
            if height and width and image.size != (width, height):
                return f"size {image.size} != audited {(width, height)}"
    except Exception as error:  # noqa: BLE001 - reported per file
        return f"{type(error).__name__}: {error}"
    return ""


def verify_yolo_variant(
    variant_root: Path,
    cohort_csv: Path,
    labels_csv: Path | None = None,
    workers: int = 4,
    decode_images: bool = True,
    report_path: Path | None = None,
) -> dict[str, object]:
    """Prove a built variant is exactly the frozen Milestone 2 cohort before training.

    Checks: no test directory, exact image and label file sets per split, label contents
    (against the frozen table when given), box totals, and that every PNG fully decodes as
    8-bit grayscale at its audited size. Writes ``verification.json`` and raises on any
    problem.
    """
    cohort = pd.read_csv(cohort_csv)
    frozen = load_frozen_labels(labels_csv) if labels_csv is not None else None
    problems: list[str] = []
    for forbidden in ("test", "testing"):
        if (variant_root / "images" / forbidden).exists() or (
            variant_root / "labels" / forbidden
        ).exists():
            problems.append(f"forbidden '{forbidden}' directory present")
    counts: dict[str, dict[str, int]] = {}
    image_tasks: list[tuple[str, int, int]] = []
    for split, group in cohort.groupby("m2_split", sort=True):
        expected_ids = set(group["image_id"].astype(str))
        image_dir = variant_root / "images" / str(split)
        label_dir = variant_root / "labels" / str(split)
        image_ids = {p.stem for p in image_dir.glob("*.png")} if image_dir.is_dir() else set()
        label_ids = {p.stem for p in label_dir.glob("*.txt")} if label_dir.is_dir() else set()
        for kind, found in (("image", image_ids), ("label", label_ids)):
            missing = expected_ids - found
            extra = found - expected_ids
            if missing:
                problems.append(
                    f"{split}: {len(missing)} missing {kind}s, e.g. {sorted(missing)[:3]}"
                )
            if extra:
                problems.append(
                    f"{split}: {len(extra)} unexpected {kind}s, e.g. {sorted(extra)[:3]}"
                )
        leftovers = [p.name for p in image_dir.glob(".*.tmp")] if image_dir.is_dir() else []
        if leftovers:
            problems.append(f"{split}: {len(leftovers)} interrupted temporary files")
        boxes = 0
        for record in group.itertuples(index=False):
            image_id = str(record.image_id)
            label_path = label_dir / f"{image_id}.txt"
            if image_id not in label_ids:
                continue
            text = label_path.read_text(encoding="utf-8")
            try:
                count = _validate_yolo_text(text, str(label_path))
            except ValueError as error:
                problems.append(str(error))
                continue
            if count != int(record.fracture_box_count):
                problems.append(
                    f"{image_id}: {count} boxes, manifest says {record.fracture_box_count}"
                )
            if frozen is not None and text != frozen[image_id]:
                problems.append(f"{image_id}: label differs from frozen labels")
            boxes += count
            if image_id in image_ids:
                image_tasks.append(
                    (
                        str(image_dir / f"{image_id}.png"),
                        int(getattr(record, "height", 0) or 0),
                        int(getattr(record, "width", 0) or 0),
                    )
                )
        counts[str(split)] = {"images": len(image_ids), "labels": len(label_ids), "boxes": boxes}
    decode_failures: list[str] = []
    if decode_images and image_tasks:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for task, error in zip(
                image_tasks, executor.map(_verify_image, image_tasks, chunksize=32), strict=True
            ):
                if error:
                    decode_failures.append(f"{Path(task[0]).name}: {error}")
    if decode_failures:
        problems.append(f"{len(decode_failures)} images failed decoding: {decode_failures[:3]}")
    report: dict[str, object] = {
        "variant_root": str(variant_root),
        "cohort": str(cohort_csv),
        "frozen_labels": str(labels_csv) if labels_csv is not None else None,
        "images_decoded": len(image_tasks) if decode_images else 0,
        "counts": counts,
        "passed": not problems,
        "problems": problems[:50],
        "problem_count": len(problems),
    }
    report_path = report_path or variant_root / "verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if problems:
        raise RuntimeError(f"Variant verification failed for {variant_root}: {problems[:5]}")
    return report


def summarize_yolo_variant(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "images": int(len(frame)),
        "splits": frame["m2_split"].value_counts().to_dict(),
        "fracture_boxes": int(frame["box_count"].sum()),
    }
