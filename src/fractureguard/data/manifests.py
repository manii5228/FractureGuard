from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECTION_NAMES = {1: "frontal", 2: "lateral", 3: "oblique"}
PASCAL_NAME_PATTERN = re.compile(rb"<name>\s*([^<]+?)\s*</name>", re.IGNORECASE)


def _flag(value: object) -> bool:
    if pd.isna(value) or value == "":
        return False
    if isinstance(value, (int, float)):
        return float(value) == 1.0
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _nullable_text(value: object) -> str | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    return str(value).strip()


def read_pascal_objects(path: Path) -> Counter[str]:
    names = [
        match.decode("utf-8").strip().lower()
        for match in PASCAL_NAME_PATTERN.findall(path.read_bytes())
    ]
    return Counter(names)


def build_grazpedwri_manifest(
    dataset_root: Path,
    output_csv: Path,
    shard_index: int = 0,
    shard_count: int = 1,
) -> pd.DataFrame:
    metadata_path = dataset_root / "dataset.csv"
    annotation_root = dataset_root / "pascalvoc"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing GRAZPEDWRI metadata: {metadata_path}")
    if not annotation_root.is_dir():
        raise FileNotFoundError(f"Missing GRAZPEDWRI annotations: {annotation_root}")

    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard index/count")
    metadata = pd.read_csv(metadata_path)
    metadata = metadata.iloc[shard_index::shard_count]
    rows: list[dict[str, object]] = []
    for record in metadata.to_dict(orient="records"):
        image_id = str(record["filestem"])
        annotation_path = annotation_root / f"{image_id}.xml"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing annotation for {image_id}: {annotation_path}")
        objects = read_pascal_objects(annotation_path)
        fracture_count = int(objects.get("fracture", 0))
        metadata_fracture = _flag(record.get("fracture_visible"))
        patient_id = int(record["patient_id"])
        study_number = int(record["study_number"])
        rows.append(
            {
                "dataset_name": "grazpedwri",
                "image_id": image_id,
                "patient_id": f"graz-{patient_id:04d}",
                "study_id": f"graz-{patient_id:04d}-{study_number:02d}",
                "image_path": f"images/{image_id}.png",
                "image_available": (dataset_root / "images" / f"{image_id}.png").is_file(),
                "projection_code": int(record["projection"]),
                "projection": PROJECTION_NAMES.get(int(record["projection"]), "unknown"),
                "laterality": str(record["laterality"]),
                "age": float(record["age"]),
                "sex": str(record["gender"]),
                "initial_exam": _flag(record.get("initial_exam")),
                "ao_classification": _nullable_text(record.get("ao_classification")),
                "diagnosis_uncertain": _flag(record.get("diagnosis_uncertain")),
                "cast": _flag(record.get("cast")),
                "metal": _flag(record.get("metal")),
                "osteopenia": _flag(record.get("osteopenia")),
                "fracture_visible_metadata": metadata_fracture,
                "fracture_visible": fracture_count > 0,
                "fracture_label_mismatch": metadata_fracture != (fracture_count > 0),
                "fracture_box_count": fracture_count,
                "annotation_object_count": int(sum(objects.values())),
                "annotation_classes_json": json.dumps(dict(sorted(objects.items()))),
                "label_path": f"pascalvoc/{image_id}.xml",
                "pixel_spacing": float(record["pixel_spacing"]),
                "device_manufacturer": str(record["device_manufacturer"]),
            }
        )

    frame = pd.DataFrame(rows).sort_values("image_id", kind="stable")
    if frame["image_id"].duplicated().any():
        raise ValueError("GRAZPEDWRI metadata contains duplicate image IDs")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def build_fracatlas_manifest(
    dataset_root: Path,
    image_audit_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    metadata_path = dataset_root / "dataset.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing FracAtlas metadata: {metadata_path}")
    if not image_audit_csv.is_file():
        raise FileNotFoundError(f"Missing FracAtlas image audit: {image_audit_csv}")

    metadata = pd.read_csv(metadata_path)
    audit = pd.read_csv(image_audit_csv)
    audit = audit.drop(columns=["image_id"], errors="ignore")
    audit["image_filename"] = audit["image_path"].map(lambda value: Path(value).name)
    audit["folder_label"] = audit["image_path"].map(lambda value: Path(value).parent.name.lower())
    audit["fractured_folder"] = audit["folder_label"].eq("fractured")

    expected = metadata[["image_id", "fractured"]].copy()
    expected["fractured"] = expected["fractured"].astype(bool)
    joined = expected.merge(
        audit,
        left_on=["image_id", "fractured"],
        right_on=["image_filename", "fractured_folder"],
        how="left",
        validate="one_to_one",
    )
    missing = joined[joined["image_path"].isna()]["image_id"].tolist()
    if missing:
        raise ValueError(f"Missing canonical FracAtlas images: {missing[:5]}")

    canonical_audit = joined[
        [
            "image_id",
            "image_path",
            "width",
            "height",
            "image_mode",
            "source_bit_depth",
            "file_size_bytes",
            "file_hash_sha256",
            "read_error",
        ]
    ].copy()
    frame = metadata.merge(canonical_audit, on="image_id", validate="one_to_one")

    hash_counts = frame["file_hash_sha256"].value_counts()
    canonical_id_by_hash = (
        frame.sort_values("image_id", kind="stable")
        .drop_duplicates("file_hash_sha256")
        .set_index("file_hash_sha256")["image_id"]
    )
    frame["dataset_name"] = "fracatlas"
    frame["patient_id"] = pd.NA
    frame["study_id"] = pd.NA
    frame["patient_level_split_possible"] = False
    frame["anatomy_group"] = frame.apply(_fracatlas_anatomy_group, axis=1)
    frame["wrist_compatible"] = pd.NA
    frame["wrist_compatibility_status"] = frame["hand"].map(
        {1: "requires_manual_anatomy_review", 0: "out_of_scope_anatomy"}
    )
    frame["projection"] = frame.apply(_fracatlas_projection, axis=1)
    frame["fracture_visible"] = frame["fractured"].astype(bool)
    frame["fracture_box_count"] = frame["fracture_count"].astype(int)
    frame["label_path"] = frame["image_id"].map(
        lambda value: f"Annotations/PASCAL VOC/{Path(value).stem}.xml"
    )
    frame["duplicate_group_size"] = frame["file_hash_sha256"].map(hash_counts).astype(int)
    frame["duplicate_canonical_image_id"] = frame["file_hash_sha256"].map(canonical_id_by_hash)
    frame["is_duplicate_copy"] = frame["image_id"] != frame["duplicate_canonical_image_id"]
    frame = frame.sort_values("image_id", kind="stable")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def merge_dataset_manifests(input_csvs: list[Path], output_csv: Path) -> pd.DataFrame:
    if not input_csvs:
        raise ValueError("At least one manifest shard is required")
    frames = [pd.read_csv(path) for path in input_csvs]
    frame = pd.concat(frames, ignore_index=True)
    if frame["image_id"].duplicated().any():
        raise ValueError("Manifest shards contain duplicate image IDs")
    frame = frame.sort_values("image_id", kind="stable")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def join_graz_image_audit(
    assigned_manifest_csv: Path,
    image_audit_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    manifest = pd.read_csv(assigned_manifest_csv)
    audit = pd.read_csv(image_audit_csv)
    if manifest["image_id"].duplicated().any() or audit["image_id"].duplicated().any():
        raise ValueError("Image IDs must be unique before joining the GRAZ image audit")
    if audit["read_error"].notna().any():
        raise ValueError("The GRAZ image audit contains unreadable files")

    audit_columns = audit[
        [
            "image_id",
            "image_path",
            "width",
            "height",
            "image_mode",
            "source_bit_depth",
            "file_size_bytes",
            "file_hash_sha256",
        ]
    ].rename(columns={"image_path": "audited_image_path"})
    frame = manifest.drop(columns=["image_available"], errors="ignore").merge(
        audit_columns,
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    missing = frame.loc[frame["file_hash_sha256"].isna(), "image_id"].tolist()
    if missing:
        raise ValueError(f"GRAZ images missing from the audit: {missing[:5]}")
    if len(frame) != len(manifest) or len(audit) != len(manifest):
        raise ValueError(
            "GRAZ manifest and image audit counts differ: "
            f"manifest={len(manifest)}, audit={len(audit)}"
        )

    hash_counts = frame["file_hash_sha256"].value_counts()
    canonical_id_by_hash = (
        frame.sort_values("image_id", kind="stable")
        .drop_duplicates("file_hash_sha256")
        .set_index("file_hash_sha256")["image_id"]
    )
    split_counts_by_hash = frame.groupby("file_hash_sha256")["split"].nunique()
    frame["duplicate_group_size"] = frame["file_hash_sha256"].map(hash_counts).astype(int)
    frame["duplicate_canonical_image_id"] = frame["file_hash_sha256"].map(canonical_id_by_hash)
    frame["is_duplicate_copy"] = frame["image_id"] != frame["duplicate_canonical_image_id"]
    frame["duplicate_crosses_split"] = frame["file_hash_sha256"].map(split_counts_by_hash).gt(1)
    frame["image_available"] = True
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def summarize_ready_graz_manifest(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "images": int(len(frame)),
        "read_errors": 0,
        "duplicate_copies": int(frame["is_duplicate_copy"].sum()),
        "duplicates_crossing_splits": int(frame["duplicate_crosses_split"].sum()),
        "width_range": [int(frame["width"].min()), int(frame["width"].max())],
        "height_range": [int(frame["height"].min()), int(frame["height"].max())],
        "image_modes": frame["image_mode"].value_counts().to_dict(),
    }


def _fracatlas_anatomy_group(row: pd.Series) -> str:
    labels = [name for name in ("hand", "leg", "hip", "shoulder") if int(row[name]) == 1]
    if int(row["mixed"]) == 1 or len(labels) != 1:
        return "mixed"
    return labels[0]


def _fracatlas_projection(row: pd.Series) -> str:
    labels = [name for name in ("frontal", "lateral", "oblique") if int(row[name]) == 1]
    return "+".join(labels) if labels else "unknown"


def summarize_dataset_manifest(frame: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {
        "rows": int(len(frame)),
        "fracture_positive": int(frame["fracture_visible"].astype(bool).sum()),
        "fracture_negative": int((~frame["fracture_visible"].astype(bool)).sum()),
    }
    if "patient_id" in frame and frame["patient_id"].notna().any():
        result["patients"] = int(frame["patient_id"].nunique())
    if "study_id" in frame and frame["study_id"].notna().any():
        result["studies"] = int(frame["study_id"].nunique())
    if "image_available" in frame:
        result["images_available"] = int(frame["image_available"].astype(bool).sum())
    if "fracture_label_mismatch" in frame:
        result["fracture_label_mismatches"] = int(
            frame["fracture_label_mismatch"].astype(bool).sum()
        )
    if "is_duplicate_copy" in frame:
        result["duplicate_copies_to_exclude"] = int(frame["is_duplicate_copy"].astype(bool).sum())
    if "anatomy_group" in frame:
        result["anatomy_groups"] = frame["anatomy_group"].value_counts().to_dict()
    return result
