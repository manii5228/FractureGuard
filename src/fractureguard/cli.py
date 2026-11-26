from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fractureguard.calibration import (
    apply_triage_policy,
    calibration_metrics,
    calibration_run_summary,
    choose_abstention_thresholds,
    fit_validation_platt_calibrator,
    reliability_table,
    selective_risk_coverage,
)
from fractureguard.config import load_dataset_config
from fractureguard.data.audit import (
    audit_images,
    exclude_audited_paths,
    iter_image_paths,
    merge_audit_files,
    select_shard,
    summarize_audit,
)
from fractureguard.data.figshare import (
    download_figshare_file,
    get_article,
    load_inventory,
    save_inventory,
    select_inventory_files,
)
from fractureguard.data.manifests import (
    build_fracatlas_manifest,
    build_grazpedwri_manifest,
    join_graz_image_audit,
    merge_dataset_manifests,
    summarize_dataset_manifest,
    summarize_ready_graz_manifest,
)
from fractureguard.data.splits import summarize_splits, write_patient_splits
from fractureguard.data.training_data import (
    HASH_CHECK_MODES,
    PREPROCESSING_VARIANTS,
    build_yolo_variant,
    create_milestone2_cohort,
    summarize_milestone2_cohort,
    summarize_yolo_variant,
    verify_yolo_variant,
)
from fractureguard.data.visual_audit import (
    create_graz_overlay_sheets,
    create_yolo_variant_comparison,
)
from fractureguard.evaluation import (
    apply_frozen_m5_policy,
    assess_fracatlas_external_test,
    build_m5_challenge_table,
    build_m5_error_case_table,
    build_m5_study_predictions,
    build_m5_subgroup_table,
    build_m5_test_cohort,
    build_m5_view_prediction_table,
    patient_bootstrap_intervals,
    summarize_m5_cohort,
    summarize_m5_error_cases,
    summarize_m5_triage,
)
from fractureguard.fusion import (
    aggregate_study_predictions,
    build_view_prediction_table,
    compare_fusion_methods,
    summarize_study_predictions,
)
from fractureguard.paths import DATA_ROOT, MANIFEST_ROOT, REPORT_ROOT, ensure_project_directories
from fractureguard.training import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fractureguard",
        description="FractureGuard data inventory and audit tools",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser(
        "figshare-inventory",
        help="Fetch Figshare article metadata without downloading dataset archives",
    )
    inventory_parser.add_argument("--dataset", required=True)
    inventory_parser.add_argument("--output", type=Path)

    download_parser = subparsers.add_parser(
        "figshare-download",
        help="Download selected files from a saved official Figshare inventory",
    )
    download_parser.add_argument("--dataset", required=True)
    download_parser.add_argument("--include", nargs="+", required=True)
    download_parser.add_argument("--destination", type=Path)

    audit_parser = subparsers.add_parser(
        "audit-images",
        help="Hash and inspect extracted image files",
    )
    audit_parser.add_argument("--root", type=Path, required=True)
    audit_parser.add_argument("--dataset", required=True)
    audit_parser.add_argument("--output", type=Path)
    audit_parser.add_argument("--shard-index", type=int, default=0)
    audit_parser.add_argument("--shard-count", type=int, default=1)
    audit_parser.add_argument(
        "--exclude-audit",
        type=Path,
        help="Skip image paths already recorded in an existing audit CSV",
    )

    merge_parser = subparsers.add_parser(
        "merge-audits",
        help="Merge non-overlapping generic image-audit shard CSV files",
    )
    merge_parser.add_argument("--input", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)

    graz_parser = subparsers.add_parser(
        "build-graz-manifest",
        help="Build the normalized GRAZPEDWRI metadata and annotation manifest",
    )
    graz_parser.add_argument("--root", type=Path, default=DATA_ROOT / "raw" / "grazpedwri-dx")
    graz_parser.add_argument(
        "--output", type=Path, default=MANIFEST_ROOT / "grazpedwri_manifest.csv"
    )
    graz_parser.add_argument("--shard-index", type=int, default=0)
    graz_parser.add_argument("--shard-count", type=int, default=1)

    frac_parser = subparsers.add_parser(
        "build-fracatlas-manifest",
        help="Build a canonical FracAtlas manifest joined to its image audit",
    )
    frac_parser.add_argument(
        "--root", type=Path, default=DATA_ROOT / "raw" / "fracatlas" / "FracAtlas"
    )
    frac_parser.add_argument(
        "--image-audit",
        type=Path,
        default=MANIFEST_ROOT / "fracatlas_generic_image_audit.csv",
    )
    frac_parser.add_argument(
        "--output", type=Path, default=MANIFEST_ROOT / "fracatlas_manifest.csv"
    )

    manifest_merge_parser = subparsers.add_parser(
        "merge-manifests",
        help="Merge non-overlapping normalized dataset-manifest shards",
    )
    manifest_merge_parser.add_argument("--input", type=Path, nargs="+", required=True)
    manifest_merge_parser.add_argument("--output", type=Path, required=True)

    split_parser = subparsers.add_parser(
        "create-graz-splits",
        help="Create frozen, fracture-stratified patient-level GRAZPEDWRI splits",
    )
    split_parser.add_argument(
        "--manifest", type=Path, default=MANIFEST_ROOT / "grazpedwri_manifest.csv"
    )
    split_parser.add_argument(
        "--patient-output", type=Path, default=DATA_ROOT / "splits" / "grazpedwri_patients.csv"
    )
    split_parser.add_argument(
        "--image-output", type=Path, default=DATA_ROOT / "splits" / "grazpedwri_images.csv"
    )
    split_parser.add_argument("--seed", type=int, default=42)

    finalize_parser = subparsers.add_parser(
        "finalize-graz-manifest",
        help="Join the GRAZ image audit to frozen split assignments and check duplicates",
    )
    finalize_parser.add_argument(
        "--manifest", type=Path, default=DATA_ROOT / "splits" / "grazpedwri_images.csv"
    )
    finalize_parser.add_argument(
        "--image-audit",
        type=Path,
        default=MANIFEST_ROOT / "grazpedwri_generic_image_audit.csv",
    )
    finalize_parser.add_argument(
        "--output", type=Path, default=MANIFEST_ROOT / "grazpedwri_ready_manifest.csv"
    )

    overlay_parser = subparsers.add_parser(
        "create-graz-overlay-audit",
        help="Create deterministic stratified GRAZ fracture-box QA sheets",
    )
    overlay_parser.add_argument(
        "--manifest", type=Path, default=MANIFEST_ROOT / "grazpedwri_ready_manifest.csv"
    )
    overlay_parser.add_argument("--root", type=Path, default=DATA_ROOT / "raw" / "grazpedwri-dx")
    overlay_parser.add_argument(
        "--output-dir", type=Path, default=Path("reports") / "figures" / "audit"
    )
    overlay_parser.add_argument("--count", type=int, default=100)
    overlay_parser.add_argument("--seed", type=int, default=42)

    cohort_parser = subparsers.add_parser(
        "prepare-m2-cohort",
        help="Freeze the leakage-safe Milestone 2 train and clean validation cohort",
    )
    cohort_parser.add_argument(
        "--manifest", type=Path, default=MANIFEST_ROOT / "grazpedwri_ready_manifest.csv"
    )
    cohort_parser.add_argument(
        "--output", type=Path, default=MANIFEST_ROOT / "grazpedwri_m2_cohort.csv"
    )

    yolo_parser = subparsers.add_parser(
        "build-yolo-variant",
        help="Create a fracture-only 8-bit YOLO dataset for one preprocessing candidate",
    )
    yolo_parser.add_argument(
        "--cohort", type=Path, default=MANIFEST_ROOT / "grazpedwri_m2_cohort.csv"
    )
    yolo_parser.add_argument("--root", type=Path, default=DATA_ROOT / "raw" / "grazpedwri-dx")
    yolo_parser.add_argument(
        "--output-root", type=Path, default=DATA_ROOT / "derived" / "grazpedwri_yolo"
    )
    yolo_parser.add_argument("--variant", choices=PREPROCESSING_VARIANTS, required=True)
    yolo_parser.add_argument("--workers", type=int, default=4)
    yolo_parser.add_argument("--limit-per-split", type=int)
    yolo_parser.add_argument(
        "--labels",
        type=Path,
        default=MANIFEST_ROOT / "grazpedwri_m2_labels.csv",
        help="Frozen fracture-only labels; pass 'none' to read source YOLOv5 labels",
    )
    yolo_parser.add_argument("--hash-check", choices=HASH_CHECK_MODES, default="strict")

    verify_parser = subparsers.add_parser(
        "verify-yolo-variant",
        help="Prove a built variant exactly matches the frozen Milestone 2 cohort",
    )
    verify_parser.add_argument("--variant-root", type=Path, required=True)
    verify_parser.add_argument(
        "--cohort", type=Path, default=MANIFEST_ROOT / "grazpedwri_m2_cohort.csv"
    )
    verify_parser.add_argument(
        "--labels", type=Path, default=MANIFEST_ROOT / "grazpedwri_m2_labels.csv"
    )
    verify_parser.add_argument("--workers", type=int, default=4)

    comparison_parser = subparsers.add_parser(
        "create-yolo-variant-comparison",
        help="Render the same validation cases across all preprocessing candidates",
    )
    comparison_parser.add_argument(
        "--cohort", type=Path, default=MANIFEST_ROOT / "grazpedwri_m2_cohort.csv"
    )
    comparison_parser.add_argument(
        "--variants-root", type=Path, default=DATA_ROOT / "derived" / "grazpedwri_yolo"
    )
    comparison_parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports") / "figures" / "audit" / "m2_preprocessing_comparison.jpg",
    )

    train_parser = subparsers.add_parser(
        "run-experiment",
        help="Run one controlled Ultralytics experiment and write its experiment card",
    )
    train_parser.add_argument("--spec", type=Path, required=True)
    train_parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu")
    train_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start a new run instead of skipping a completed or resuming an interrupted one",
    )

    view_table_parser = subparsers.add_parser(
        "build-m3-view-table",
        help="Join per-box detector outputs to the frozen train/validation manifest",
    )
    view_table_parser.add_argument(
        "--manifest", type=Path, default=MANIFEST_ROOT / "grazpedwri_m2_cohort.csv"
    )
    view_table_parser.add_argument(
        "--detections",
        type=Path,
        required=True,
        help="CSV with one predicted fracture box per row: image_id,confidence",
    )
    view_table_parser.add_argument(
        "--output", type=Path, default=Path("reports") / "predictions" / "m3_views.csv"
    )

    study_table_parser = subparsers.add_parser(
        "build-m3-study-table",
        help="Create AP/PA, lateral, maximum, mean, and rule-based study scores",
    )
    study_table_parser.add_argument("--views", type=Path, required=True)
    study_table_parser.add_argument(
        "--output", type=Path, default=Path("reports") / "predictions" / "m3_studies.csv"
    )
    study_table_parser.add_argument("--agreement-threshold", type=float, default=0.35)
    study_table_parser.add_argument("--high-confidence", type=float, default=0.75)
    study_table_parser.add_argument("--multi-box-threshold", type=int, default=2)

    fusion_compare_parser = subparsers.add_parser(
        "compare-m3-fusion",
        help="Compare single-view and fused validation study metrics at one frozen threshold",
    )
    fusion_compare_parser.add_argument("--studies", type=Path, required=True)
    fusion_compare_parser.add_argument(
        "--output", type=Path, default=Path("reports") / "tables" / "m3_fusion_comparison.csv"
    )
    fusion_compare_parser.add_argument("--threshold", type=float, default=0.5)

    calibration_parser = subparsers.add_parser(
        "calibrate-m4",
        help="Fit validation-only Platt calibration and freeze an abstention policy",
    )
    calibration_parser.add_argument("--studies", type=Path, required=True)
    calibration_parser.add_argument(
        "--output-dir", type=Path, default=REPORT_ROOT / "predictions" / "m4"
    )
    calibration_parser.add_argument("--score-column", default="fused_max_score")
    calibration_parser.add_argument("--label-column", default="study_fracture_visible")
    calibration_parser.add_argument("--reliability-bins", type=int, default=10)
    calibration_parser.add_argument(
        "--maximum-negative-risk",
        type=float,
        default=0.05,
        help="Maximum validation fracture fraction in the reviewer-screened no-fracture queue",
    )
    calibration_parser.add_argument("--minimum-positive-precision", type=float, default=0.98)
    calibration_parser.add_argument(
        "--no-plot", action="store_true", help="Write CSV/JSON artefacts without the PNG figure"
    )

    m5_cohort_parser = subparsers.add_parser(
        "prepare-m5-test-cohort",
        help="Freeze the clean internal-test cohort after the M4 policy is frozen",
    )
    m5_cohort_parser.add_argument(
        "--manifest", type=Path, default=MANIFEST_ROOT / "grazpedwri_ready_manifest.csv"
    )
    m5_cohort_parser.add_argument(
        "--output", type=Path, default=MANIFEST_ROOT / "grazpedwri_m5_test_cohort.csv"
    )

    m5_views_parser = subparsers.add_parser(
        "build-m5-view-table", help="Join test-only detector boxes to the frozen M5 cohort"
    )
    m5_views_parser.add_argument(
        "--manifest", type=Path, default=MANIFEST_ROOT / "grazpedwri_m5_test_cohort.csv"
    )
    m5_views_parser.add_argument("--detections", type=Path, required=True)
    m5_views_parser.add_argument(
        "--output", type=Path, default=REPORT_ROOT / "predictions" / "m5" / "m5_views.csv"
    )

    m5_studies_parser = subparsers.add_parser(
        "build-m5-study-table", help="Create unfitted test-only fused study scores"
    )
    m5_studies_parser.add_argument("--views", type=Path, required=True)
    m5_studies_parser.add_argument(
        "--output", type=Path, default=REPORT_ROOT / "predictions" / "m5" / "m5_studies_raw.csv"
    )

    m5_evaluate_parser = subparsers.add_parser(
        "evaluate-m5", help="Apply the frozen M4 policy and patient-bootstrap internal-test metrics"
    )
    m5_evaluate_parser.add_argument("--studies", type=Path, required=True)
    m5_evaluate_parser.add_argument(
        "--policy",
        type=Path,
        default=REPORT_ROOT / "predictions" / "m4" / "m4_frozen_triage_policy.json",
    )
    m5_evaluate_parser.add_argument(
        "--output-dir", type=Path, default=REPORT_ROOT / "predictions" / "m5"
    )
    m5_evaluate_parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    m5_evaluate_parser.add_argument("--seed", type=int, default=42)

    m5_analysis_parser = subparsers.add_parser(
        "analyze-m5", help="Create frozen-policy subgroup, challenge, and error-analysis tables"
    )
    m5_analysis_parser.add_argument("--studies", type=Path, required=True)
    m5_analysis_parser.add_argument(
        "--cohort", type=Path, default=MANIFEST_ROOT / "grazpedwri_m5_test_cohort.csv"
    )
    m5_analysis_parser.add_argument(
        "--output-dir", type=Path, default=REPORT_ROOT / "predictions" / "m5"
    )

    fracatlas_gate_parser = subparsers.add_parser(
        "audit-m5-fracatlas", help="Gate FracAtlas external testing on explicit wrist labels"
    )
    fracatlas_gate_parser.add_argument(
        "--manifest", type=Path, default=MANIFEST_ROOT / "fracatlas_manifest.csv"
    )
    fracatlas_gate_parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_ROOT / "predictions" / "m5" / "m5_fracatlas_eligibility.json",
    )

    degradation_parser = subparsers.add_parser(
        "evaluate-m5-degradations",
        help="Apply the frozen policy to detector boxes from fixed degradation conditions",
    )
    degradation_parser.add_argument(
        "--manifest", type=Path, default=MANIFEST_ROOT / "grazpedwri_m5_test_cohort.csv"
    )
    degradation_parser.add_argument("--detections-dir", type=Path, required=True)
    degradation_parser.add_argument(
        "--policy",
        type=Path,
        default=REPORT_ROOT / "predictions" / "m4" / "m4_frozen_triage_policy.json",
    )
    degradation_parser.add_argument(
        "--output-dir", type=Path, default=REPORT_ROOT / "predictions" / "m5"
    )

    return parser


def run_figshare_inventory(dataset_key: str, output: Path | None) -> int:
    dataset = load_dataset_config(dataset_key)
    article_id = dataset.get("figshare_article_id")
    if article_id is None:
        raise ValueError(f"Dataset '{dataset_key}' has no Figshare article ID")
    destination = output or MANIFEST_ROOT / "inventory" / f"{dataset_key}_figshare.json"
    article = get_article(int(article_id))
    inventory = save_inventory(article, dataset_key=dataset_key, output_path=destination)
    size_gib = inventory["total_size_bytes"] / (1024**3)
    print(
        json.dumps(
            {
                "dataset": dataset_key,
                "title": inventory["title"],
                "file_count": len(inventory["files"]),
                "total_size_gib": round(size_gib, 3),
                "output": str(destination),
            },
            indent=2,
        )
    )
    return 0


def run_audit(
    root: Path,
    dataset_key: str,
    output: Path | None,
    shard_index: int,
    shard_count: int,
    exclude_audit: Path | None,
) -> int:
    if output is None and shard_count == 1:
        destination = MANIFEST_ROOT / f"{dataset_key}_generic_image_audit.csv"
    elif output is None:
        destination = MANIFEST_ROOT / (
            f"{dataset_key}_generic_image_audit_part_{shard_index + 1:03d}_of_{shard_count:03d}.csv"
        )
    else:
        destination = output
    all_paths = list(iter_image_paths(root.resolve()))
    if exclude_audit is not None:
        all_paths = exclude_audited_paths(all_paths, root.resolve(), exclude_audit)
    shard_paths = select_shard(all_paths, shard_index, shard_count)
    frame = audit_images(
        root=root.resolve(),
        dataset_name=dataset_key,
        output_csv=destination,
        paths=shard_paths,
    )
    print(json.dumps(summarize_audit(frame), indent=2, default=str))
    print(f"Manifest: {destination}")
    return 0


def run_merge_audits(inputs: list[Path], output: Path) -> int:
    frame = merge_audit_files(inputs, output)
    print(json.dumps(summarize_audit(frame), indent=2, default=str))
    print(f"Merged manifest: {output}")
    return 0


def run_figshare_download(
    dataset_key: str,
    include: list[str],
    destination: Path | None,
) -> int:
    inventory_path = MANIFEST_ROOT / "inventory" / f"{dataset_key}_figshare.json"
    dataset = load_dataset_config(dataset_key)
    article_id = dataset.get("figshare_article_id")
    if article_id is None:
        raise ValueError(f"Dataset '{dataset_key}' has no Figshare article ID")
    # Figshare file URLs are short-lived signed URLs. Refresh the official
    # article record immediately before downloading while retaining the saved
    # inventory as a reproducibility artifact.
    article = get_article(int(article_id))
    save_inventory(article, dataset_key=dataset_key, output_path=inventory_path)
    inventory = load_inventory(inventory_path)
    selected = select_inventory_files(inventory, include)
    destination_dir = destination or DATA_ROOT / "cache" / "downloads" / dataset_key
    results = [download_figshare_file(item, destination_dir) for item in selected]
    print(json.dumps(results, indent=2))
    return 0


def _write_reliability_plot(table: pd.DataFrame, output: Path) -> None:
    """Render the reliability figure only when the optional ML plotting stack is installed."""
    import matplotlib.pyplot as plt

    populated = table.loc[table["count"] > 0]
    figure, axis = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    axis.plot([0.0, 1.0], [0.0, 1.0], "--", color="0.5", label="perfect calibration")
    marker_sizes = 35 + 8 * populated["count"].pow(0.5)
    axis.scatter(
        populated["mean_probability"],
        populated["observed_frequency"],
        s=marker_sizes,
        color="#1769aa",
        label="validation bins",
    )
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), xlabel="Calibrated probability")
    axis.set_ylabel("Observed fracture frequency")
    axis.set_title("Milestone 4 reliability — validation only")
    axis.legend(loc="upper left")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_m4_calibration(
    studies_path: Path,
    output_dir: Path,
    *,
    score_column: str,
    label_column: str,
    reliability_bins: int,
    maximum_negative_risk: float,
    minimum_positive_precision: float,
    render_plot: bool,
) -> int:
    """Fit calibration and save validation-only artefacts for the frozen M3 maximum score."""
    studies = pd.read_csv(studies_path)
    calibrator = fit_validation_platt_calibrator(
        studies, score_column=score_column, label_column=label_column
    )
    probabilities = calibrator.predict(pd.to_numeric(studies[score_column], errors="raise"))
    metrics = calibration_metrics(
        studies[label_column], probabilities, bins=reliability_bins
    )
    reliability = reliability_table(studies[label_column], probabilities, bins=reliability_bins)
    policy = choose_abstention_thresholds(
        probabilities,
        studies[label_column],
        maximum_negative_risk=maximum_negative_risk,
        minimum_positive_precision=minimum_positive_precision,
    )
    calibrated = apply_triage_policy(
        studies,
        probabilities,
        negative_threshold=float(policy["negative_threshold"]),
        positive_threshold=float(policy["positive_threshold"]),
    )
    risk_coverage = selective_risk_coverage(probabilities, studies[label_column])

    output_dir.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(output_dir / "m4_studies_calibrated.csv", index=False)
    reliability.to_csv(output_dir / "m4_reliability.csv", index=False)
    risk_coverage.to_csv(output_dir / "m4_risk_coverage.csv", index=False)
    summary = calibration_run_summary(calibrator, metrics, policy)
    summary.update(
        {
            "policy_status": "frozen_reviewer_only_prototype_policy",
            "release_rule": (
                "Every model result, including no_fracture_detected, requires reviewer approval "
                "before it may be released to a patient/demo user."
            ),
            "input_studies": str(studies_path),
            "score_column": score_column,
            "label_column": label_column,
            "outputs": {
                "calibrated_studies": "m4_studies_calibrated.csv",
                "reliability_table": "m4_reliability.csv",
                "risk_coverage_table": "m4_risk_coverage.csv",
                "frozen_policy": "m4_frozen_triage_policy.json",
            },
        }
    )
    with (output_dir / "m4_calibration_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "m4_platt_calibrator.json").open("w", encoding="utf-8") as handle:
        json.dump(calibrator.to_dict(), handle, indent=2)
    frozen_policy = {
        "status": summary["policy_status"],
        "scope": summary["scope"],
        "release_rule": summary["release_rule"],
        "score_column": score_column,
        "calibrator": calibrator.to_dict(),
        "negative_threshold": policy["negative_threshold"],
        "positive_threshold": policy["positive_threshold"],
        "validation_characteristics": policy,
    }
    with (output_dir / "m4_frozen_triage_policy.json").open("w", encoding="utf-8") as handle:
        json.dump(frozen_policy, handle, indent=2)
    if render_plot:
        plot_path = output_dir / "m4_reliability.png"
        _write_reliability_plot(reliability, plot_path)
        summary["outputs"]["reliability_plot"] = plot_path.name
        with (output_dir / "m4_calibration_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


def run_m5_evaluation(
    studies_path: Path,
    policy_path: Path,
    output_dir: Path,
    *,
    bootstrap_repetitions: int,
    seed: int,
) -> int:
    """Apply the already frozen M4 policy and report test metrics without refitting."""
    triaged = apply_frozen_m5_policy(pd.read_csv(studies_path), policy_path)
    summary = summarize_m5_triage(triaged)
    intervals = patient_bootstrap_intervals(
        triaged, repetitions=bootstrap_repetitions, seed=seed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    triaged.to_csv(output_dir / "m5_studies_triaged.csv", index=False)
    intervals.to_csv(output_dir / "m5_patient_bootstrap_intervals.csv", index=False)
    report = {
        "scope": "untouched_internal_test",
        "policy": str(policy_path),
        "input_studies": str(studies_path),
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_seed": seed,
        "metrics": summary,
    }
    with (output_dir / "m5_test_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0


def run_m5_analysis(studies_path: Path, cohort_path: Path, output_dir: Path) -> int:
    """Write reproducible fixed-policy subgroup, challenge, and case-level analyses."""
    studies = pd.read_csv(studies_path)
    cohort = pd.read_csv(cohort_path)
    subgroups = build_m5_subgroup_table(studies, cohort)
    challenges = build_m5_challenge_table(studies, cohort)
    cases = build_m5_error_case_table(studies, cohort)
    case_summary = summarize_m5_error_cases(cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "subgroups": output_dir / "m5_subgroup_metrics.csv",
        "challenge_cohorts": output_dir / "m5_challenge_cohort_metrics.csv",
        "error_cases": output_dir / "m5_error_cases.csv",
        "error_summary": output_dir / "m5_error_summary.csv",
    }
    subgroups.to_csv(outputs["subgroups"], index=False)
    challenges.to_csv(outputs["challenge_cohorts"], index=False)
    cases.to_csv(outputs["error_cases"], index=False)
    case_summary.to_csv(outputs["error_summary"], index=False)
    report = {
        "scope": "frozen_policy_internal_test_secondary_analyses",
        "input_studies": str(studies_path),
        "input_cohort": str(cohort_path),
        "subgroup_rows": len(subgroups),
        "challenge_rows": len(challenges),
        "error_and_review_cases": len(cases),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    with (output_dir / "m5_secondary_analysis_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0


def run_m5_degradation_evaluation(
    manifest_path: Path, detections_dir: Path, policy_path: Path, output_dir: Path
) -> int:
    """Evaluate every fixed degradation export without refitting the frozen policy."""
    manifest = pd.read_csv(manifest_path)
    detection_paths = sorted(detections_dir.glob("m5_detector_boxes_*.csv"))
    if not detection_paths:
        raise FileNotFoundError(f"No M5 degradation detector CSVs found in {detections_dir}")
    view_tables: list[pd.DataFrame] = []
    study_tables: list[pd.DataFrame] = []
    triaged_tables: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    prefix = "m5_detector_boxes_"
    for path in detection_paths:
        condition = path.stem.removeprefix(prefix)
        views = build_m5_view_prediction_table(manifest, pd.read_csv(path))
        studies = build_m5_study_predictions(views)
        triaged = apply_frozen_m5_policy(studies, policy_path)
        views.insert(0, "degradation", condition)
        studies.insert(0, "degradation", condition)
        triaged.insert(0, "degradation", condition)
        view_tables.append(views)
        study_tables.append(studies)
        triaged_tables.append(triaged)
        metric_rows.append({"degradation": condition, **summarize_m5_triage(triaged)})

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "views": output_dir / "m5_degradation_views.csv",
        "studies_raw": output_dir / "m5_degradation_studies_raw.csv",
        "studies_triaged": output_dir / "m5_degradation_studies_triaged.csv",
        "metrics": output_dir / "m5_degradation_metrics.csv",
    }
    pd.concat(view_tables, ignore_index=True).to_csv(outputs["views"], index=False)
    pd.concat(study_tables, ignore_index=True).to_csv(outputs["studies_raw"], index=False)
    pd.concat(triaged_tables, ignore_index=True).to_csv(outputs["studies_triaged"], index=False)
    metrics = pd.DataFrame.from_records(metric_rows)
    metrics.to_csv(outputs["metrics"], index=False)
    report = {
        "scope": "fixed_artificial_degradations_on_internal_test",
        "policy": str(policy_path),
        "conditions": metrics.to_dict(orient="records"),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    with (output_dir / "m5_degradation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    ensure_project_directories()
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "figshare-inventory":
        return run_figshare_inventory(args.dataset, args.output)
    if args.command == "figshare-download":
        return run_figshare_download(args.dataset, args.include, args.destination)
    if args.command == "audit-images":
        return run_audit(
            args.root,
            args.dataset,
            args.output,
            args.shard_index,
            args.shard_count,
            args.exclude_audit,
        )
    if args.command == "merge-audits":
        return run_merge_audits(args.input, args.output)
    if args.command == "build-graz-manifest":
        frame = build_grazpedwri_manifest(
            args.root,
            args.output,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        print(json.dumps(summarize_dataset_manifest(frame), indent=2))
        print(f"Manifest: {args.output}")
        return 0
    if args.command == "build-fracatlas-manifest":
        frame = build_fracatlas_manifest(args.root, args.image_audit, args.output)
        print(json.dumps(summarize_dataset_manifest(frame), indent=2))
        print(f"Manifest: {args.output}")
        return 0
    if args.command == "merge-manifests":
        frame = merge_dataset_manifests(args.input, args.output)
        print(json.dumps(summarize_dataset_manifest(frame), indent=2))
        print(f"Merged manifest: {args.output}")
        return 0
    if args.command == "create-graz-splits":
        patients, images = write_patient_splits(
            args.manifest,
            args.patient_output,
            args.image_output,
            seed=args.seed,
        )
        print(json.dumps(summarize_splits(patients, images), indent=2))
        print(f"Patient splits: {args.patient_output}")
        print(f"Image assignments: {args.image_output}")
        return 0
    if args.command == "finalize-graz-manifest":
        frame = join_graz_image_audit(args.manifest, args.image_audit, args.output)
        print(json.dumps(summarize_ready_graz_manifest(frame), indent=2))
        print(f"Ready manifest: {args.output}")
        return 0
    if args.command == "create-graz-overlay-audit":
        outputs = create_graz_overlay_sheets(
            args.manifest,
            args.root,
            args.output_dir,
            count=args.count,
            seed=args.seed,
        )
        print(
            json.dumps({"cases": args.count, "sheets": [str(path) for path in outputs]}, indent=2)
        )
        return 0
    if args.command == "prepare-m2-cohort":
        frame = create_milestone2_cohort(args.manifest, args.output)
        print(json.dumps(summarize_milestone2_cohort(frame), indent=2))
        print(f"Milestone 2 cohort: {args.output}")
        return 0
    if args.command == "build-yolo-variant":
        frame = build_yolo_variant(
            args.cohort,
            args.root,
            args.output_root,
            args.variant,
            workers=args.workers,
            limit_per_split=args.limit_per_split,
            labels_csv=None if str(args.labels).lower() == "none" else args.labels,
            hash_check=args.hash_check,
        )
        print(json.dumps(summarize_yolo_variant(frame), indent=2))
        return 0
    if args.command == "verify-yolo-variant":
        report = verify_yolo_variant(
            args.variant_root, args.cohort, args.labels, workers=args.workers
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "create-yolo-variant-comparison":
        output = create_yolo_variant_comparison(
            args.cohort,
            args.variants_root,
            args.output,
        )
        print(f"Comparison sheet: {output}")
        return 0
    if args.command == "run-experiment":
        result = run_experiment(args.spec, device=args.device, resume=not args.fresh)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "build-m3-view-table":
        table = build_view_prediction_table(
            pd.read_csv(args.manifest), pd.read_csv(args.detections)
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(json.dumps({"rows": len(table), "output": str(args.output)}, indent=2))
        return 0
    if args.command == "build-m3-study-table":
        table = aggregate_study_predictions(
            pd.read_csv(args.views),
            agreement_threshold=args.agreement_threshold,
            high_confidence=args.high_confidence,
            multi_box_threshold=args.multi_box_threshold,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(
            json.dumps(
                {**summarize_study_predictions(table), "output": str(args.output)}, indent=2
            )
        )
        return 0
    if args.command == "compare-m3-fusion":
        table = compare_fusion_methods(pd.read_csv(args.studies), threshold=args.threshold)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(json.dumps({"rows": len(table), "output": str(args.output)}, indent=2))
        return 0
    if args.command == "calibrate-m4":
        return run_m4_calibration(
            args.studies,
            args.output_dir,
            score_column=args.score_column,
            label_column=args.label_column,
            reliability_bins=args.reliability_bins,
            maximum_negative_risk=args.maximum_negative_risk,
            minimum_positive_precision=args.minimum_positive_precision,
            render_plot=not args.no_plot,
        )
    if args.command == "prepare-m5-test-cohort":
        cohort = build_m5_test_cohort(pd.read_csv(args.manifest))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cohort.to_csv(args.output, index=False)
        print(json.dumps({**summarize_m5_cohort(cohort), "output": str(args.output)}, indent=2))
        return 0
    if args.command == "build-m5-view-table":
        table = build_m5_view_prediction_table(
            pd.read_csv(args.manifest), pd.read_csv(args.detections)
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(json.dumps({"rows": len(table), "output": str(args.output)}, indent=2))
        return 0
    if args.command == "build-m5-study-table":
        table = build_m5_study_predictions(pd.read_csv(args.views))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(json.dumps({"rows": len(table), "output": str(args.output)}, indent=2))
        return 0
    if args.command == "evaluate-m5":
        return run_m5_evaluation(
            args.studies,
            args.policy,
            args.output_dir,
            bootstrap_repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        )
    if args.command == "analyze-m5":
        return run_m5_analysis(args.studies, args.cohort, args.output_dir)
    if args.command == "audit-m5-fracatlas":
        report = assess_fracatlas_external_test(pd.read_csv(args.manifest))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps({**report, "output": str(args.output)}, indent=2))
        return 0
    if args.command == "evaluate-m5-degradations":
        return run_m5_degradation_evaluation(
            args.manifest, args.detections_dir, args.policy, args.output_dir
        )
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
