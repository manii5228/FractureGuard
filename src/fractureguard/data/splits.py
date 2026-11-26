from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


def _allocate_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    exact = {name: total * fraction for name, fraction in fractions.items()}
    counts = {name: int(value) for name, value in exact.items()}
    remaining = total - sum(counts.values())
    order = sorted(fractions, key=lambda name: (exact[name] - counts[name], name), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def create_patient_splits(
    manifest: pd.DataFrame,
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")
    if manifest["patient_id"].isna().any():
        raise ValueError("Patient-level splitting requires a patient ID for every row")

    patient_labels = (
        manifest.groupby("patient_id", as_index=False)["fracture_visible"]
        .max()
        .rename(columns={"fracture_visible": "ever_fracture_positive"})
    )
    assignments: list[dict[str, object]] = []
    for label, group in patient_labels.groupby("ever_fracture_positive", sort=True):
        patient_ids = sorted(group["patient_id"].astype(str).tolist())
        random.Random(seed + int(bool(label))).shuffle(patient_ids)
        counts = _allocate_counts(len(patient_ids), fractions)
        offset = 0
        for split_name in ("train", "validation", "test"):
            end = offset + counts[split_name]
            assignments.extend(
                {
                    "patient_id": patient_id,
                    "split": split_name,
                    "ever_fracture_positive": bool(label),
                    "split_seed": seed,
                }
                for patient_id in patient_ids[offset:end]
            )
            offset = end

    patients = pd.DataFrame(assignments).sort_values("patient_id", kind="stable")
    images = manifest.merge(
        patients[["patient_id", "split"]], on="patient_id", validate="many_to_one"
    )
    if len(images) != len(manifest):
        raise ValueError("Split join changed the image count")
    if patients["patient_id"].duplicated().any():
        raise ValueError("A patient was assigned to more than one split")
    return patients, images


def write_patient_splits(
    manifest_csv: Path,
    patient_output_csv: Path,
    image_output_csv: Path,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(manifest_csv)
    patients, images = create_patient_splits(manifest, seed=seed)
    patient_output_csv.parent.mkdir(parents=True, exist_ok=True)
    image_output_csv.parent.mkdir(parents=True, exist_ok=True)
    patients.to_csv(patient_output_csv, index=False)
    images.to_csv(image_output_csv, index=False)
    return patients, images


def summarize_splits(patients: pd.DataFrame, images: pd.DataFrame) -> dict[str, object]:
    patient_table = pd.crosstab(patients["split"], patients["ever_fracture_positive"])
    image_table = pd.crosstab(images["split"], images["fracture_visible"])
    return {
        "patients_by_split_and_ever_positive": {
            str(index): {str(column): int(value) for column, value in row.items()}
            for index, row in patient_table.iterrows()
        },
        "images_by_split_and_positive": {
            str(index): {str(column): int(value) for column, value in row.items()}
            for index, row in image_table.iterrows()
        },
        "patient_overlap_count": int(patients.groupby("patient_id")["split"].nunique().gt(1).sum()),
    }
