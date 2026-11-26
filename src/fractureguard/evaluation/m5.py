"""Milestone 5 evaluation that permits only the frozen internal-test partition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fractureguard.calibration import PlattCalibrator, apply_triage_policy
from fractureguard.fusion import aggregate_study_predictions, build_view_prediction_table

SUPPORTED_PROJECTIONS = frozenset({"frontal", "lateral"})
MIN_SUBGROUP_STUDIES = 30
MIN_SUBGROUP_CLASS_STUDIES = 10


def _bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0).ne(0)
    return values.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _normalise_split(value: object) -> str:
    normalised = str(value).strip().lower()
    return {"validation": "val", "testing": "test"}.get(normalised, normalised)


def build_m5_test_cohort(ready_manifest: pd.DataFrame) -> pd.DataFrame:
    """Freeze the clean internal-test cohort matching the M2 validation inclusion policy."""
    required = {
        "image_id", "patient_id", "study_id", "split", "projection", "initial_exam",
        "diagnosis_uncertain", "cast", "metal", "fracture_visible",
    }
    missing = required.difference(ready_manifest.columns)
    if missing:
        raise ValueError(f"Ready manifest missing columns: {sorted(missing)}")
    frame = ready_manifest.copy()
    test = frame["split"].map(_normalise_split).eq("test")
    clean = (
        _bool_series(frame["initial_exam"])
        & ~_bool_series(frame["diagnosis_uncertain"])
        & ~_bool_series(frame["cast"])
        & ~_bool_series(frame["metal"])
        & frame["projection"].isin(SUPPORTED_PROJECTIONS)
    )
    cohort = frame.loc[test & clean].copy()
    if cohort.empty:
        raise ValueError("No clean internal-test images were selected")
    if cohort["image_id"].astype(str).duplicated().any():
        raise ValueError("Internal-test cohort contains duplicate image IDs")
    cohort["m5_split"] = "test"
    return cohort.sort_values(["patient_id", "study_id", "image_id"], kind="stable").reset_index(
        drop=True
    )


def summarize_m5_cohort(cohort: pd.DataFrame) -> dict[str, int]:
    labels = _bool_series(cohort["fracture_visible"])
    return {
        "images": int(len(cohort)),
        "patients": int(cohort["patient_id"].nunique()),
        "studies": int(cohort["study_id"].nunique()),
        "positive_images": int(labels.sum()),
        "positive_studies": int(
            cohort.assign(_label=labels).groupby("study_id")["_label"].any().sum()
        ),
    }


def build_m5_view_prediction_table(
    manifest: pd.DataFrame, detections: pd.DataFrame
) -> pd.DataFrame:
    """Create view predictions from test-only detector boxes; train/validation rows are refused."""
    frame = manifest.copy()
    split_column = "m5_split" if "m5_split" in frame.columns else "split"
    if (
        split_column not in frame.columns
        or not frame[split_column].map(_normalise_split).eq("test").all()
    ):
        raise ValueError("Milestone 5 view inputs must be test-only")
    frame["m2_split"] = "test"
    return build_view_prediction_table(frame, detections, allowed_splits=frozenset({"test"}))


def build_m5_study_predictions(views: pd.DataFrame) -> pd.DataFrame:
    """Create raw fused internal-test study scores without calibrating or selecting anything."""
    if not views["fusion_split"].map(_normalise_split).eq("test").all():
        raise ValueError("Milestone 5 study inputs must be test-only")
    return aggregate_study_predictions(views, allowed_splits=frozenset({"test"}))


def _load_policy(policy_path: Path) -> tuple[dict[str, object], PlattCalibrator]:
    with policy_path.open(encoding="utf-8") as handle:
        policy = json.load(handle)
    if policy.get("status") != "frozen_reviewer_only_prototype_policy":
        raise ValueError("Milestone 5 requires the frozen reviewer-only M4 policy")
    parameters = policy.get("calibrator")
    if not isinstance(parameters, dict) or parameters.get("method") != "platt_logistic":
        raise ValueError("Frozen policy does not contain a Platt calibrator")
    calibrator = PlattCalibrator(
        coefficient=float(parameters["coefficient"]),
        intercept=float(parameters["intercept"]),
        iterations=int(parameters["iterations"]),
        negative_log_likelihood=float(parameters["negative_log_likelihood"]),
        l2_penalty=float(parameters["l2_penalty"]),
    )
    return policy, calibrator


def apply_frozen_m5_policy(studies: pd.DataFrame, policy_path: Path) -> pd.DataFrame:
    """Apply the saved M4 transform and thresholds unchanged to test-only study scores."""
    if (
        "fusion_split" not in studies.columns
        or not studies["fusion_split"].map(_normalise_split).eq("test").all()
    ):
        raise ValueError("Milestone 5 evaluation requires test-only fused study predictions")
    policy, calibrator = _load_policy(policy_path)
    score_column = str(policy["score_column"])
    if score_column not in studies.columns:
        raise ValueError(f"Test studies missing frozen policy score column: {score_column}")
    probabilities = calibrator.predict(pd.to_numeric(studies[score_column], errors="raise"))
    return apply_triage_policy(
        studies,
        probabilities,
        negative_threshold=float(policy["negative_threshold"]),
        positive_threshold=float(policy["positive_threshold"]),
    )


def summarize_m5_triage(studies: pd.DataFrame) -> dict[str, float | int | None]:
    required = {"patient_id", "study_fracture_visible", "calibrated_probability", "triage_decision"}
    missing = required.difference(studies.columns)
    if missing:
        raise ValueError(f"Triaged test studies missing columns: {sorted(missing)}")
    labels = _bool_series(studies["study_fracture_visible"])
    no_fracture = studies["triage_decision"].eq("no_fracture_detected")
    suspected = studies["triage_decision"].eq("suspected_fracture")
    uncertain = studies["triage_decision"].eq("uncertain_manual_review")
    no_count = int(no_fracture.sum())
    suspected_count = int(suspected.sum())
    uncertain_count = int(uncertain.sum())
    false_negative = int((no_fracture & labels).sum())
    false_positive = int((suspected & ~labels).sum())
    accepted = no_count + suspected_count
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    return {
        "studies": int(len(studies)),
        "positive_studies": positive_count,
        "negative_studies": negative_count,
        "no_fracture_detected_count": no_count,
        "suspected_fracture_count": suspected_count,
        "uncertain_manual_review_count": uncertain_count,
        "automatic_coverage": accepted / len(studies),
        "review_rate": uncertain_count / len(studies),
        "negative_false_negative_count": false_negative,
        "negative_risk": false_negative / no_count if no_count else None,
        "positive_false_positive_count": false_positive,
        "positive_precision": float(labels[suspected].mean()) if suspected_count else None,
        "accepted_error_count": false_negative + false_positive,
        "accepted_error_rate": (false_negative + false_positive) / accepted if accepted else None,
        "suspected_triage_sensitivity": (
            float((suspected & labels).sum() / positive_count) if positive_count else None
        ),
        "no_fracture_triage_specificity": (
            float((no_fracture & ~labels).sum() / negative_count) if negative_count else None
        ),
    }


def attach_m5_study_metadata(studies: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Attach one deterministic metadata row to each frozen test study."""
    required_studies = {
        "study_id", "patient_id", "frontal_available", "lateral_available",
        "frontal_score", "lateral_score", "frontal_box_count", "lateral_box_count",
    }
    missing = required_studies.difference(studies.columns)
    if missing:
        raise ValueError(f"M5 studies missing columns: {sorted(missing)}")
    required_cohort = {"study_id", "patient_id", "age", "sex", "device_manufacturer"}
    missing = required_cohort.difference(cohort.columns)
    if missing:
        raise ValueError(f"M5 cohort missing metadata columns: {sorted(missing)}")

    frame = cohort.copy()
    for column in ("patient_id", "age", "sex"):
        inconsistent = frame.groupby("study_id")[column].nunique(dropna=True)
        if (inconsistent > 1).any():
            raise ValueError(f"A study_id maps to multiple {column} values")

    def first_or_unknown(values: pd.Series) -> object:
        present = values.dropna()
        return present.iloc[0] if not present.empty else "unknown"

    def joined_values(values: pd.Series) -> str:
        present = sorted({str(value).strip() for value in values.dropna() if str(value).strip()})
        return "+".join(present) if present else "unknown"

    aggregations: dict[str, tuple[str, object]] = {
        "metadata_patient_id": ("patient_id", first_or_unknown),
        "age": ("age", first_or_unknown),
        "sex": ("sex", first_or_unknown),
        "device_manufacturer": ("device_manufacturer", joined_values),
    }
    for column in ("fracture_box_count", "annotation_object_count"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
            aggregations[f"metadata_{column}"] = (column, "sum")
    for column in ("width", "height"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            aggregations[f"minimum_{column}"] = (column, "min")

    metadata = frame.groupby("study_id", as_index=False).agg(**aggregations)
    metadata["study_id"] = metadata["study_id"].astype(str)
    result = studies.copy()
    result["study_id"] = result["study_id"].astype(str)
    result = result.merge(metadata, on="study_id", how="left", validate="one_to_one")
    if result["metadata_patient_id"].isna().any():
        raise ValueError("M5 cohort is missing one or more evaluated study IDs")
    if not result["patient_id"].astype(str).eq(result["metadata_patient_id"].astype(str)).all():
        raise ValueError("Patient IDs disagree between the M5 cohort and study predictions")

    age = pd.to_numeric(result["age"], errors="coerce")
    result["age_band"] = pd.cut(
        age,
        bins=[-np.inf, 7, 11, 15, np.inf],
        labels=["under_8", "8_to_11", "12_to_15", "16_or_older"],
    ).astype("string").fillna("unknown")
    result["sex"] = result["sex"].astype(str).str.strip().str.upper().replace({"": "UNKNOWN"})
    result["device_manufacturer"] = (
        result["device_manufacturer"].astype(str).str.strip().replace({"": "unknown"})
    )
    frontal = _bool_series(result["frontal_available"])
    lateral = _bool_series(result["lateral_available"])
    result["view_configuration"] = np.select(
        [frontal & lateral, frontal, lateral],
        ["paired", "frontal_only", "lateral_only"],
        default="unsupported",
    )
    return result.drop(columns="metadata_patient_id")


def _analysis_row(
    subset: pd.DataFrame, *, analysis: str, group: str, value: str
) -> dict[str, object]:
    metrics = summarize_m5_triage(subset)
    eligible = (
        int(metrics["studies"]) >= MIN_SUBGROUP_STUDIES
        and int(metrics["positive_studies"]) >= MIN_SUBGROUP_CLASS_STUDIES
        and int(metrics["negative_studies"]) >= MIN_SUBGROUP_CLASS_STUDIES
    )
    return {
        "analysis": analysis,
        "group": group,
        "value": value,
        "eligible_for_comparative_claim": eligible,
        **metrics,
    }


def build_m5_subgroup_table(studies: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Report prespecified demographic/acquisition subgroups without changing the policy."""
    enriched = attach_m5_study_metadata(studies, cohort)
    rows: list[dict[str, object]] = []
    for group in ("sex", "age_band", "view_configuration", "device_manufacturer"):
        for value, subset in enriched.groupby(group, dropna=False, sort=True):
            rows.append(
                _analysis_row(subset, analysis="subgroup", group=group, value=str(value))
            )
    return pd.DataFrame.from_records(rows)


def build_m5_challenge_table(studies: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Evaluate difficult, fixed cohorts; small cohorts remain descriptive only."""
    enriched = attach_m5_study_metadata(studies, cohort)
    frontal = pd.to_numeric(enriched["frontal_score"], errors="coerce")
    lateral = pd.to_numeric(enriched["lateral_score"], errors="coerce")
    total_boxes = (
        pd.to_numeric(enriched["frontal_box_count"], errors="coerce").fillna(0)
        + pd.to_numeric(enriched["lateral_box_count"], errors="coerce").fillna(0)
    )
    missing_dimension = pd.Series(np.nan, index=enriched.index)
    minimum_dimension = pd.concat(
        [
            pd.to_numeric(enriched.get("minimum_width", missing_dimension), errors="coerce"),
            pd.to_numeric(enriched.get("minimum_height", missing_dimension), errors="coerce"),
        ],
        axis=1,
    ).min(axis=1)
    masks = {
        "single_view": enriched["view_configuration"].ne("paired"),
        "paired_score_disagreement_ge_0_25": (
            enriched["view_configuration"].eq("paired") & frontal.sub(lateral).abs().ge(0.25)
        ),
        "no_predicted_boxes": total_boxes.eq(0),
        "multiple_predicted_boxes": total_boxes.ge(2),
        "source_min_dimension_under_1000": minimum_dimension.lt(1000),
    }
    metadata_boxes = enriched.get("metadata_fracture_box_count")
    if metadata_boxes is not None:
        masks["multiple_annotated_fracture_boxes"] = pd.to_numeric(
            metadata_boxes, errors="coerce"
        ).fillna(0).ge(2)

    rows = [
        _analysis_row(
            enriched.loc[mask], analysis="challenge", group="challenge_cohort", value=name
        )
        for name, mask in masks.items()
        if bool(mask.any())
    ]
    return pd.DataFrame.from_records(rows)


def build_m5_error_case_table(studies: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Return all accepted-policy errors plus review cases for auditable case analysis."""
    enriched = attach_m5_study_metadata(studies, cohort)
    labels = _bool_series(enriched["study_fracture_visible"])
    decisions = enriched["triage_decision"].astype(str)
    conditions = [
        decisions.eq("no_fracture_detected") & labels,
        decisions.eq("suspected_fracture") & ~labels,
        decisions.eq("uncertain_manual_review") & labels,
        decisions.eq("uncertain_manual_review") & ~labels,
    ]
    categories = [
        "accepted_false_negative",
        "accepted_false_positive",
        "manual_review_positive",
        "manual_review_negative",
    ]
    enriched["case_category"] = np.select(conditions, categories, default="correct_automatic")
    selected = enriched.loc[enriched["case_category"].ne("correct_automatic")].copy()
    preferred = [
        "study_id", "patient_id", "study_fracture_visible", "triage_decision",
        "case_category", "calibrated_probability", "fused_max_score", "frontal_score",
        "lateral_score", "frontal_box_count", "lateral_box_count", "age", "age_band",
        "sex", "device_manufacturer", "view_configuration",
        "metadata_fracture_box_count", "minimum_width", "minimum_height",
    ]
    columns = [column for column in preferred if column in selected.columns]
    return selected.loc[:, columns].sort_values(
        ["case_category", "calibrated_probability"], kind="stable"
    ).reset_index(drop=True)


def summarize_m5_error_cases(cases: pd.DataFrame) -> pd.DataFrame:
    """Summarize structured error/review cases by category and acquisition context."""
    if "case_category" not in cases.columns:
        raise ValueError("M5 error cases missing case_category")
    dimensions = [
        column
        for column in ("case_category", "view_configuration", "sex", "age_band")
        if column in cases.columns
    ]
    rows: list[dict[str, object]] = []
    for dimension in dimensions:
        for value, subset in cases.groupby(dimension, dropna=False, sort=True):
            rows.append(
                {
                    "dimension": dimension,
                    "value": str(value),
                    "cases": int(len(subset)),
                    "positive_cases": int(_bool_series(subset["study_fracture_visible"]).sum()),
                    "median_probability": float(
                        pd.to_numeric(subset["calibrated_probability"], errors="coerce").median()
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def assess_fracatlas_external_test(manifest: pd.DataFrame) -> dict[str, object]:
    """Check whether FracAtlas contains a defensible, explicitly labelled wrist test subset."""
    required = {"image_id", "anatomy_group", "fracture_visible", "wrist_compatible"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"FracAtlas manifest missing columns: {sorted(missing)}")
    frame = manifest.copy()
    wrist_raw = frame["wrist_compatible"]
    normalised = wrist_raw.astype("string").str.strip().str.lower()
    explicit = wrist_raw.notna() & normalised.isin({"true", "false", "1", "0", "yes", "no"})
    wrist = normalised.isin({"true", "1", "yes"})
    hand = frame["anatomy_group"].astype(str).str.strip().str.lower().eq("hand")
    unique = ~_bool_series(frame.get("is_duplicate_copy", pd.Series(False, index=frame.index)))
    eligible = explicit & wrist & hand & unique
    labels = _bool_series(frame.loc[eligible, "fracture_visible"])
    patient_ids_available = bool(
        eligible.any()
        and "patient_id" in frame.columns
        and frame.loc[eligible, "patient_id"].notna().all()
    )
    ready = bool(eligible.any() and labels.any() and (~labels).any())
    status = "ready_image_level_only" if ready else "blocked_manual_anatomy_review"
    reason = (
        "Explicit wrist-compatible labels are present; evaluate only at image level because "
        "FracAtlas has no reliable patient grouping."
        if ready
        else "Hand images are not equivalent to wrist radiographs. Explicit wrist-compatible "
        "labels are missing, so an untouched wrist test would be scientifically invalid."
    )
    return {
        "status": status,
        "reason": reason,
        "total_images": int(len(frame)),
        "hand_images": int(hand.sum()),
        "explicit_wrist_compatibility_labels": int(explicit.sum()),
        "eligible_unique_wrist_images": int(eligible.sum()),
        "eligible_positive_images": int(labels.sum()),
        "eligible_negative_images": int((~labels).sum()) if len(labels) else 0,
        "patient_ids_available": patient_ids_available,
        "required_action": (
            None
            if ready
            else "Create and independently verify wrist_compatible labels before inference."
        ),
    }


def patient_bootstrap_intervals(
    studies: pd.DataFrame, *, repetitions: int = 2000, seed: int = 42, confidence: float = 0.95
) -> pd.DataFrame:
    """Patient-resample confidence intervals for fixed-policy triage metrics."""
    if repetitions < 100 or not 0.0 < confidence < 1.0:
        raise ValueError("repetitions must be at least 100 and confidence must be between 0 and 1")
    patients = studies["patient_id"].astype(str).unique()
    if len(patients) < 2:
        raise ValueError("Patient bootstrap requires at least two patients")
    grouped = studies.assign(_patient=studies["patient_id"].astype(str)).groupby("_patient")
    groups = {patient: group for patient, group in grouped}
    point = summarize_m5_triage(studies)
    keys = [
        "automatic_coverage",
        "review_rate",
        "negative_risk",
        "positive_precision",
        "accepted_error_rate",
        "suspected_triage_sensitivity",
        "no_fracture_triage_specificity",
    ]
    draws = {key: [] for key in keys}
    random = np.random.default_rng(seed)
    for _ in range(repetitions):
        selected_patients = random.choice(patients, size=len(patients), replace=True)
        sample = [groups[patient] for patient in selected_patients]
        result = summarize_m5_triage(pd.concat(sample, ignore_index=True))
        for key in keys:
            value = result.get(key)
            if value is not None:
                draws[key].append(float(value))
    alpha = (1.0 - confidence) / 2.0
    return pd.DataFrame(
        {
            "metric": keys,
            "point_estimate": [point.get(key) for key in keys],
            "ci_lower": [
                float(np.quantile(draws[key], alpha)) if draws[key] else np.nan for key in keys
            ],
            "ci_upper": [
                float(np.quantile(draws[key], 1.0 - alpha)) if draws[key] else np.nan
                for key in keys
            ],
            "bootstrap_repetitions": repetitions,
            "resampling_unit": "patient",
        }
    )
