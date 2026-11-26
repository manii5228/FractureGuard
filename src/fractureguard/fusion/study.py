"""Validation-only AP/PA and lateral study-fusion utilities.

Detector exports are expected to contain one row per predicted fracture box with the
columns ``image_id`` and ``confidence``. Images without boxes are deliberately added from
the manifest with a score of zero; otherwise a detector's false negatives would disappear
from the study-level evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

SUPPORTED_PROJECTIONS = frozenset({"frontal", "lateral"})
REQUIRED_MANIFEST_COLUMNS = frozenset(
    {"image_id", "patient_id", "study_id", "projection", "fracture_visible"}
)
REQUIRED_DETECTION_COLUMNS = frozenset({"image_id", "confidence"})


def _normalise_split(value: object) -> str:
    normalised = str(value).strip().lower()
    aliases = {"validation": "val", "valid": "val", "training": "train"}
    return aliases.get(normalised, normalised)


def _bool_series(values: pd.Series) -> pd.Series:
    """Parse CSV-safe boolean values without treating the string ``False`` as true."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    true_values = {"1", "true", "t", "yes", "y"}
    return values.fillna(False).map(lambda value: str(value).strip().lower() in true_values)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _prepare_manifest(
    manifest: pd.DataFrame, *, allowed_splits: frozenset[str] = frozenset({"train", "val"})
) -> pd.DataFrame:
    _require_columns(manifest, REQUIRED_MANIFEST_COLUMNS, "Manifest")
    frame = manifest.copy()
    if frame["image_id"].isna().any() or frame["image_id"].astype(str).duplicated().any():
        raise ValueError("Manifest image_id values must be present and unique")

    split_column = "m2_split" if "m2_split" in frame.columns else "split"
    if split_column not in frame.columns:
        raise ValueError("Manifest must contain either m2_split or split")
    frame["fusion_split"] = frame[split_column].map(_normalise_split)
    source_split_is_test = (
        frame["split"].map(_normalise_split).eq("test").any()
        if "split" in frame.columns
        else False
    )
    if (
        (frame["fusion_split"].eq("test").any() or source_split_is_test)
        and "test" not in allowed_splits
    ):
        raise ValueError("Fusion inputs must not contain test rows")
    unsupported_splits = set(frame["fusion_split"]).difference(allowed_splits)
    if unsupported_splits:
        raise ValueError(
            f"Fusion supports only train/val rows, found: {sorted(unsupported_splits)}"
        )

    frame["image_id"] = frame["image_id"].astype(str)
    frame["projection"] = frame["projection"].astype(str).str.strip().str.lower()
    frame["fracture_visible"] = _bool_series(frame["fracture_visible"])
    return frame.loc[frame["projection"].isin(SUPPORTED_PROJECTIONS)].copy()


def _aggregate_detections(detections: pd.DataFrame, image_ids: set[str]) -> pd.DataFrame:
    _require_columns(detections, REQUIRED_DETECTION_COLUMNS, "Detections")
    frame = detections.copy()
    frame["image_id"] = frame["image_id"].astype(str)
    unknown = set(frame["image_id"]).difference(image_ids)
    if unknown:
        sample = sorted(unknown)[:5]
        raise ValueError(f"Detections contain image_id values absent from manifest: {sample}")

    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    if frame["confidence"].isna().any() or not np.isfinite(frame["confidence"]).all():
        raise ValueError("Detection confidence must be finite numeric values")
    if ((frame["confidence"] < 0.0) | (frame["confidence"] > 1.0)).any():
        raise ValueError("Detection confidence must be between 0 and 1")
    if frame.empty:
        return pd.DataFrame(columns=["image_id", "detected_box_count", "view_score", "top3_score"])

    ordered = frame.sort_values(["image_id", "confidence"], ascending=[True, False])
    top3 = ordered.groupby("image_id", sort=False).head(3).groupby("image_id")["confidence"].mean()
    summary = frame.groupby("image_id", as_index=False).agg(
        detected_box_count=("confidence", "size"), view_score=("confidence", "max")
    )
    summary["top3_score"] = summary["image_id"].map(top3).astype(float)
    return summary


def build_view_prediction_table(
    manifest: pd.DataFrame,
    detections: pd.DataFrame,
    *,
    allowed_splits: frozenset[str] = frozenset({"train", "val"}),
) -> pd.DataFrame:
    """Attach detector boxes to every supported train/validation image.

    The input ``detections`` has one row per predicted fracture box. A missing image is a
    genuine zero-detection prediction, not missing data. Test rows and unrecognised image
    identifiers are rejected before any table is produced.
    """
    images = _prepare_manifest(manifest, allowed_splits=allowed_splits)
    if images.empty:
        raise ValueError("Manifest contains no supported frontal or lateral images")
    summaries = _aggregate_detections(detections, set(images["image_id"]))
    columns = [
        "image_id",
        "patient_id",
        "study_id",
        "fusion_split",
        "projection",
        "fracture_visible",
    ]
    table = images.loc[:, columns].merge(
        summaries, on="image_id", how="left", validate="one_to_one"
    )
    table["detected_box_count"] = table["detected_box_count"].fillna(0).astype(int)
    table["view_score"] = table["view_score"].fillna(0.0).astype(float)
    table["top3_score"] = table["top3_score"].fillna(0.0).astype(float)
    return table.sort_values(["fusion_split", "patient_id", "study_id", "image_id"]).reset_index(
        drop=True
    )


def _rule_based_score(
    frontal_score: float | None,
    lateral_score: float | None,
    frontal_boxes: int,
    lateral_boxes: int,
    *,
    agreement_threshold: float,
    high_confidence: float,
    multi_box_threshold: int,
) -> tuple[float, str]:
    scores = [score for score in (frontal_score, lateral_score) if score is not None]
    if len(scores) == 1:
        return scores[0], "single_view"
    maximum, minimum = max(scores), min(scores)
    mean = float(np.mean(scores))
    if minimum >= agreement_threshold:
        # Corroborating views raise the score modestly but never beyond one.
        return min(1.0, maximum + 0.15 * minimum), "cross_view_agreement"
    if maximum >= high_confidence:
        return maximum, "high_confidence_single_view"
    if max(frontal_boxes, lateral_boxes) >= multi_box_threshold:
        return maximum, "multiple_boxes_single_view"
    return mean, "conservative_mean"


def aggregate_study_predictions(
    views: pd.DataFrame,
    *,
    agreement_threshold: float = 0.35,
    high_confidence: float = 0.75,
    multi_box_threshold: int = 2,
    allowed_splits: frozenset[str] = frozenset({"train", "val"}),
) -> pd.DataFrame:
    """Create one study row with frontal-only and three deterministic fused scores."""
    required = {
        "image_id",
        "patient_id",
        "study_id",
        "fusion_split",
        "projection",
        "fracture_visible",
        "detected_box_count",
        "view_score",
    }
    _require_columns(views, required, "View prediction table")
    if not 0.0 <= agreement_threshold <= 1.0 or not 0.0 <= high_confidence <= 1.0:
        raise ValueError("Rule thresholds must be between 0 and 1")
    if multi_box_threshold < 1:
        raise ValueError("multi_box_threshold must be at least 1")

    frame = views.copy()
    if frame["image_id"].astype(str).duplicated().any():
        raise ValueError("View prediction table must contain one row per image")
    frame["fusion_split"] = frame["fusion_split"].map(_normalise_split)
    if frame["fusion_split"].eq("test").any() and "test" not in allowed_splits:
        raise ValueError("Fusion inputs must not contain test rows")
    if set(frame["fusion_split"]).difference(allowed_splits):
        raise ValueError(f"Fusion supports only these splits: {sorted(allowed_splits)}")
    frame["projection"] = frame["projection"].astype(str).str.strip().str.lower()
    frame = frame.loc[frame["projection"].isin(SUPPORTED_PROJECTIONS)].copy()
    if frame.empty:
        raise ValueError("View prediction table contains no supported projections")
    frame["fracture_visible"] = _bool_series(frame["fracture_visible"])
    frame["view_score"] = pd.to_numeric(frame["view_score"], errors="coerce")
    frame["detected_box_count"] = pd.to_numeric(frame["detected_box_count"], errors="coerce")
    if frame[["view_score", "detected_box_count"]].isna().any().any():
        raise ValueError("View scores and detected-box counts must be numeric")
    if ((frame["view_score"] < 0.0) | (frame["view_score"] > 1.0)).any():
        raise ValueError("View scores must be between 0 and 1")
    if (frame["detected_box_count"] < 0).any():
        raise ValueError("Detected-box counts cannot be negative")
    for column in ("patient_id", "fusion_split"):
        inconsistent = frame.groupby("study_id")[column].nunique(dropna=False)
        if (inconsistent > 1).any():
            raise ValueError(f"A study_id maps to multiple {column} values")

    records: list[dict[str, Any]] = []
    for study_id, group in frame.groupby("study_id", sort=True):
        scores: dict[str, float | None] = {}
        boxes: dict[str, int] = {}
        image_counts: dict[str, int] = {}
        for projection in sorted(SUPPORTED_PROJECTIONS):
            subset = group.loc[group["projection"] == projection]
            scores[projection] = float(subset["view_score"].max()) if not subset.empty else None
            boxes[projection] = int(subset["detected_box_count"].sum())
            image_counts[projection] = int(len(subset))
        available_scores = [score for score in scores.values() if score is not None]
        rule_score, rule_reason = _rule_based_score(
            scores["frontal"],
            scores["lateral"],
            boxes["frontal"],
            boxes["lateral"],
            agreement_threshold=agreement_threshold,
            high_confidence=high_confidence,
            multi_box_threshold=multi_box_threshold,
        )
        records.append(
            {
                "study_id": study_id,
                "patient_id": str(group["patient_id"].iloc[0]),
                "fusion_split": str(group["fusion_split"].iloc[0]),
                "study_fracture_visible": bool(group["fracture_visible"].any()),
                "frontal_available": scores["frontal"] is not None,
                "lateral_available": scores["lateral"] is not None,
                "frontal_image_count": image_counts["frontal"],
                "lateral_image_count": image_counts["lateral"],
                "frontal_box_count": boxes["frontal"],
                "lateral_box_count": boxes["lateral"],
                "frontal_score": scores["frontal"],
                "lateral_score": scores["lateral"],
                "fused_max_score": max(available_scores),
                "fused_mean_score": float(np.mean(available_scores)),
                "fused_rule_score": rule_score,
                "rule_reason": rule_reason,
                "rule_agreement_threshold": agreement_threshold,
                "rule_high_confidence": high_confidence,
                "rule_multi_box_threshold": multi_box_threshold,
            }
        )
    return pd.DataFrame.from_records(records).sort_values("study_id").reset_index(drop=True)


def _binary_metrics(
    labels: pd.Series, scores: pd.Series, threshold: float
) -> dict[str, float | int]:
    positive = _bool_series(labels)
    predicted = scores.astype(float).ge(threshold)
    tp = int((positive & predicted).sum())
    fn = int((positive & ~predicted).sum())
    fp = int((~positive & predicted).sum())
    tn = int((~positive & ~predicted).sum())
    sensitivity_denominator = tp + fn
    specificity_denominator = tn + fp
    return {
        "studies": int(len(labels)),
        "positive_studies": int(positive.sum()),
        "negative_studies": int((~positive).sum()),
        "true_positive": tp,
        "false_negative": fn,
        "true_negative": tn,
        "false_positive": fp,
        "sensitivity": tp / sensitivity_denominator if sensitivity_denominator else np.nan,
        "specificity": tn / specificity_denominator if specificity_denominator else np.nan,
        "accuracy": (tp + tn) / len(labels) if len(labels) else np.nan,
    }


def compare_fusion_methods(studies: pd.DataFrame, *, threshold: float = 0.5) -> pd.DataFrame:
    """Compare the defined single-view and deterministic fusion methods at one threshold."""
    required = {
        "study_fracture_visible",
        "frontal_available",
        "lateral_available",
        "frontal_score",
        "lateral_score",
        "fused_max_score",
        "fused_mean_score",
        "fused_rule_score",
    }
    _require_columns(studies, required, "Study prediction table")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    methods = (
        ("frontal_only", "frontal_score", "frontal_available"),
        ("lateral_only", "lateral_score", "lateral_available"),
        ("maximum", "fused_max_score", None),
        ("mean", "fused_mean_score", None),
        ("rule_based", "fused_rule_score", None),
    )
    labels = _bool_series(studies["study_fracture_visible"])
    rows: list[dict[str, float | int | str]] = []
    frontal_score = pd.to_numeric(studies["frontal_score"], errors="coerce")
    lateral_score = pd.to_numeric(studies["lateral_score"], errors="coerce")
    for method, score_column, availability_column in methods:
        mask = (
            _bool_series(studies[availability_column])
            if availability_column is not None
            else pd.Series(True, index=studies.index)
        )
        scores = pd.to_numeric(studies.loc[mask, score_column], errors="coerce")
        if scores.isna().any():
            raise ValueError(f"{method} has missing scores for available studies")
        metrics = _binary_metrics(labels.loc[mask], scores, threshold)
        if availability_column is None:
            fused = scores.ge(threshold)
            positive = labels.loc[mask]
            frontal_missed = frontal_score.loc[mask].isna() | frontal_score.loc[mask].lt(threshold)
            lateral_missed = lateral_score.loc[mask].isna() | lateral_score.loc[mask].lt(threshold)
            metrics["frontal_misses_rescued"] = int((positive & frontal_missed & fused).sum())
            metrics["lateral_misses_rescued"] = int((positive & lateral_missed & fused).sum())
        else:
            metrics["frontal_misses_rescued"] = np.nan
            metrics["lateral_misses_rescued"] = np.nan
        rows.append({"method": method, "threshold": threshold, **metrics})
    return pd.DataFrame.from_records(rows)


def summarize_study_predictions(studies: pd.DataFrame) -> dict[str, int]:
    """Summarise pairing and class counts without calculating test metrics."""
    _require_columns(
        studies,
        {"fusion_split", "study_fracture_visible", "frontal_available", "lateral_available"},
        "Study prediction table",
    )
    if studies["fusion_split"].map(_normalise_split).eq("test").any():
        raise ValueError("Fusion inputs must not contain test rows")
    labels = _bool_series(studies["study_fracture_visible"])
    frontal = _bool_series(studies["frontal_available"])
    lateral = _bool_series(studies["lateral_available"])
    return {
        "studies": int(len(studies)),
        "positive_studies": int(labels.sum()),
        "negative_studies": int((~labels).sum()),
        "paired_studies": int((frontal & lateral).sum()),
        "frontal_only_studies": int((frontal & ~lateral).sum()),
        "lateral_only_studies": int((~frontal & lateral).sum()),
        "train_studies": int(studies["fusion_split"].map(_normalise_split).eq("train").sum()),
        "val_studies": int(studies["fusion_split"].map(_normalise_split).eq("val").sum()),
    }
