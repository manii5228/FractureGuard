"""Validation-only calibration and abstention for study-level fracture triage.

The utilities intentionally reject train/test mixtures. Calibration and threshold selection
are allowed only on validation data, then the saved policy may be frozen for test evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


def _bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0).ne(0)
    true_values = {"1", "true", "t", "yes", "y"}
    return values.fillna(False).map(lambda value: str(value).strip().lower() in true_values)


def _normalise_split(value: object) -> str:
    aliases = {"validation": "val", "valid": "val", "training": "train"}
    normalised = str(value).strip().lower()
    return aliases.get(normalised, normalised)


def _validation_split_column(studies: pd.DataFrame) -> str:
    for column in ("fusion_split", "m2_split", "split"):
        if column in studies.columns:
            values = studies[column].map(_normalise_split)
            if values.eq("test").any():
                raise ValueError("Calibration inputs must not contain test rows")
            if not values.eq("val").all():
                found = sorted(values.dropna().unique())
                raise ValueError(
                    "Calibration requires a validation-only table; "
                    f"{column} contains {found}"
                )
            return column
    raise ValueError("Calibration requires fusion_split, m2_split, or split")


def _scores_and_labels(
    studies: pd.DataFrame, *, score_column: str, label_column: str
) -> tuple[np.ndarray, np.ndarray]:
    missing = {score_column, label_column}.difference(studies.columns)
    if missing:
        raise ValueError(f"Study table missing columns: {sorted(missing)}")
    if studies.empty:
        raise ValueError("Study table is empty")
    scores = pd.to_numeric(studies[score_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError(f"{score_column} must contain finite values between 0 and 1")
    labels = _bool_series(studies[label_column]).to_numpy(dtype=float)
    if labels.min() == labels.max():
        raise ValueError(
            "Calibration requires both fracture-positive and fracture-negative studies"
        )
    return scores, labels


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -700.0, 700.0)))


@dataclass(frozen=True)
class PlattCalibrator:
    """A serialisable logistic mapping from raw detector score to probability."""

    coefficient: float
    intercept: float
    iterations: int
    negative_log_likelihood: float
    l2_penalty: float

    def predict(self, scores: pd.Series | np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
            raise ValueError("Scores for calibration must be finite values between 0 and 1")
        return _sigmoid(self.coefficient * values + self.intercept)

    def to_dict(self) -> dict[str, float | int | str]:
        return {"method": "platt_logistic", **asdict(self)}


def fit_validation_platt_calibrator(
    studies: pd.DataFrame,
    *,
    score_column: str = "fused_max_score",
    label_column: str = "study_fracture_visible",
    l2_penalty: float = 1e-6,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> PlattCalibrator:
    """Fit a numerically stable Platt calibrator on a validation-only study table."""
    if l2_penalty <= 0.0:
        raise ValueError("l2_penalty must be positive")
    if max_iterations < 1 or tolerance <= 0.0:
        raise ValueError("max_iterations and tolerance must be positive")
    _validation_split_column(studies)
    scores, labels = _scores_and_labels(
        studies, score_column=score_column, label_column=label_column
    )

    prevalence = float(labels.mean())
    coefficient = 1.0
    intercept = float(np.log(prevalence / (1.0 - prevalence)))

    def objective(a: float, b: float) -> float:
        logits = a * scores + b
        return float(
            np.sum(np.logaddexp(0.0, logits) - labels * logits) + 0.5 * l2_penalty * a**2
        )

    loss = objective(coefficient, intercept)
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        probabilities = _sigmoid(coefficient * scores + intercept)
        residual = probabilities - labels
        gradient = np.array(
            [np.dot(residual, scores) + l2_penalty * coefficient, residual.sum()], dtype=float
        )
        weight = probabilities * (1.0 - probabilities)
        hessian = np.array(
            [
                [np.dot(weight, scores * scores) + l2_penalty, np.dot(weight, scores)],
                [np.dot(weight, scores), weight.sum()],
            ],
            dtype=float,
        )
        step = np.linalg.solve(hessian, gradient)
        scale = 1.0
        while scale >= 1e-8:
            candidate_a = coefficient - scale * step[0]
            candidate_b = intercept - scale * step[1]
            candidate_loss = objective(candidate_a, candidate_b)
            if candidate_loss <= loss:
                coefficient, intercept, loss = candidate_a, candidate_b, candidate_loss
                break
            scale *= 0.5
        else:
            break
        if float(np.max(np.abs(scale * step))) <= tolerance:
            break

    return PlattCalibrator(
        coefficient=float(coefficient),
        intercept=float(intercept),
        iterations=iterations,
        negative_log_likelihood=float(loss),
        l2_penalty=float(l2_penalty),
    )


def reliability_table(
    labels: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """Build a fixed-width reliability table; empty bins are retained for plotting."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    targets = _bool_series(pd.Series(labels)).to_numpy(dtype=float)
    predicted = np.asarray(probabilities, dtype=float)
    if len(targets) != len(predicted) or not len(targets):
        raise ValueError("Labels and probabilities must have the same non-zero length")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.digitize(predicted, edges[1:-1], right=True)
    rows: list[dict[str, float | int]] = []
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        mean_probability = float(predicted[mask].mean()) if count else np.nan
        observed_frequency = float(targets[mask].mean()) if count else np.nan
        rows.append(
            {
                "bin": index + 1,
                "lower_bound": float(edges[index]),
                "upper_bound": float(edges[index + 1]),
                "count": count,
                "fraction": count / len(targets),
                "mean_probability": mean_probability,
                "observed_frequency": observed_frequency,
                "absolute_gap": abs(mean_probability - observed_frequency) if count else 0.0,
            }
        )
    return pd.DataFrame.from_records(rows)


def calibration_metrics(
    labels: pd.Series | np.ndarray, probabilities: pd.Series | np.ndarray, *, bins: int = 10
) -> dict[str, float | int]:
    """Return Brier score and expected calibration error using fixed-width bins."""
    targets = _bool_series(pd.Series(labels)).to_numpy(dtype=float)
    predicted = np.asarray(probabilities, dtype=float)
    if len(targets) != len(predicted) or not len(targets):
        raise ValueError("Labels and probabilities must have the same non-zero length")
    if not np.isfinite(predicted).all() or ((predicted < 0.0) | (predicted > 1.0)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1")
    table = reliability_table(targets, predicted, bins=bins)
    return {
        "studies": int(len(targets)),
        "prevalence": float(targets.mean()),
        "brier_score": float(np.mean((predicted - targets) ** 2)),
        "expected_calibration_error": float((table["fraction"] * table["absolute_gap"]).sum()),
        "reliability_bins": int(bins),
    }


def triage_metrics(
    probabilities: pd.Series | np.ndarray,
    labels: pd.Series | np.ndarray,
    negative_threshold: float,
    positive_threshold: float,
) -> dict[str, float | int | None]:
    """Measure validation characteristics for a fixed three-way triage policy."""
    if not 0.0 <= negative_threshold < positive_threshold <= 1.0:
        raise ValueError("Thresholds must satisfy 0 <= negative < positive <= 1")
    predicted = np.asarray(probabilities, dtype=float)
    targets = _bool_series(pd.Series(labels)).to_numpy(dtype=bool)
    if len(predicted) != len(targets) or not len(predicted):
        raise ValueError("Labels and probabilities must have the same non-zero length")
    negative = predicted < negative_threshold
    positive = predicted >= positive_threshold
    uncertain = ~(negative | positive)
    negative_count = int(negative.sum())
    positive_count = int(positive.sum())
    uncertain_count = int(uncertain.sum())
    false_negative = int((negative & targets).sum())
    false_positive = int((positive & ~targets).sum())
    accepted_count = negative_count + positive_count
    accepted_error_count = false_negative + false_positive
    return {
        "studies": int(len(targets)),
        "no_fracture_detected_count": negative_count,
        "suspected_fracture_count": positive_count,
        "uncertain_manual_review_count": uncertain_count,
        "automatic_coverage": accepted_count / len(targets),
        "review_rate": uncertain_count / len(targets),
        "negative_false_negative_count": false_negative,
        "negative_risk": false_negative / negative_count if negative_count else None,
        "positive_false_positive_count": false_positive,
        "positive_precision": float(targets[positive].mean()) if positive_count else None,
        "accepted_error_count": accepted_error_count,
        "accepted_error_rate": accepted_error_count / accepted_count if accepted_count else None,
    }


def choose_abstention_thresholds(
    probabilities: pd.Series | np.ndarray,
    labels: pd.Series | np.ndarray,
    *,
    maximum_negative_risk: float = 0.02,
    minimum_positive_precision: float = 0.98,
) -> dict[str, float | int | None]:
    """Select validation thresholds that maximise automatic coverage under explicit limits."""
    if not 0.0 <= maximum_negative_risk < 1.0:
        raise ValueError("maximum_negative_risk must be in [0, 1)")
    if not 0.0 < minimum_positive_precision <= 1.0:
        raise ValueError("minimum_positive_precision must be in (0, 1]")
    predicted = np.asarray(probabilities, dtype=float)
    targets = _bool_series(pd.Series(labels)).to_numpy(dtype=bool)
    if len(predicted) != len(targets) or not len(predicted):
        raise ValueError("Labels and probabilities must have the same non-zero length")
    if not np.isfinite(predicted).all() or ((predicted < 0.0) | (predicted > 1.0)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1")

    candidates = np.unique(np.concatenate(([0.0], predicted, [1.0])))
    negative_threshold = 0.0
    for threshold in candidates:
        accepted = predicted < threshold
        if accepted.any() and float(targets[accepted].mean()) <= maximum_negative_risk:
            negative_threshold = float(threshold)

    positive_threshold: float | None = None
    for threshold in candidates[candidates >= negative_threshold]:
        accepted = predicted >= threshold
        if accepted.any() and float(targets[accepted].mean()) >= minimum_positive_precision:
            positive_threshold = float(threshold)
            break
    if positive_threshold is None:
        positive_threshold = 1.0
    if positive_threshold <= negative_threshold:
        positive_threshold = float(np.nextafter(negative_threshold, 1.0))

    return {
        "negative_threshold": negative_threshold,
        "positive_threshold": positive_threshold,
        "maximum_negative_risk": maximum_negative_risk,
        "minimum_positive_precision": minimum_positive_precision,
        **triage_metrics(predicted, targets, negative_threshold, positive_threshold),
    }


def apply_triage_policy(
    studies: pd.DataFrame,
    probabilities: pd.Series | np.ndarray,
    *,
    negative_threshold: float,
    positive_threshold: float,
) -> pd.DataFrame:
    """Attach calibrated probability and three-way triage decision to each study row."""
    if not 0.0 <= negative_threshold < positive_threshold <= 1.0:
        raise ValueError("Thresholds must satisfy 0 <= negative < positive <= 1")
    predicted = np.asarray(probabilities, dtype=float)
    if len(studies) != len(predicted):
        raise ValueError("One calibrated probability is required per study")
    decisions = np.where(
        predicted < negative_threshold,
        "no_fracture_detected",
        np.where(predicted >= positive_threshold, "suspected_fracture", "uncertain_manual_review"),
    )
    result = studies.copy()
    result["calibrated_probability"] = predicted
    result["triage_decision"] = decisions
    return result


def selective_risk_coverage(
    probabilities: pd.Series | np.ndarray,
    labels: pd.Series | np.ndarray,
    *,
    points: int = 50,
) -> pd.DataFrame:
    """Build a confidence-ranked selective risk/coverage curve for validation studies."""
    if points < 2:
        raise ValueError("points must be at least 2")
    predicted = np.asarray(probabilities, dtype=float)
    targets = _bool_series(pd.Series(labels)).to_numpy(dtype=bool)
    if len(predicted) != len(targets) or not len(predicted):
        raise ValueError("Labels and probabilities must have the same non-zero length")
    confidence = np.abs(predicted - 0.5) * 2.0
    errors = (predicted >= 0.5) != targets
    order = np.argsort(-confidence, kind="stable")
    rows: list[dict[str, float | int]] = []
    for accepted_count in np.unique(np.linspace(1, len(predicted), points, dtype=int)):
        selected = order[:accepted_count]
        rows.append(
            {
                "accepted_count": int(accepted_count),
                "coverage": accepted_count / len(predicted),
                "selective_error_count": int(errors[selected].sum()),
                "selective_risk": float(errors[selected].mean()),
                "minimum_confidence": float(confidence[selected].min()),
            }
        )
    return pd.DataFrame.from_records(rows)


def calibration_run_summary(
    calibrator: PlattCalibrator,
    metrics: dict[str, float | int],
    policy: dict[str, float | int | None],
) -> dict[str, Any]:
    """Return a JSON-ready, explicitly validation-only report summary."""
    return {
        "scope": "validation_only",
        "warning": (
            "Calibration, thresholds, and triage characteristics were fitted and measured on "
            "the same validation cohort. They are apparent validation results, not an "
            "independent clinical-performance claim."
        ),
        "calibrator": calibrator.to_dict(),
        "calibration_metrics": metrics,
        "abstention_policy": policy,
    }
