from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from fractureguard.paths import PROJECT_ROOT, REPORT_ROOT


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_experiment_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    if not isinstance(spec, dict):
        raise ValueError(f"Expected a mapping in experiment spec: {path}")
    required = {"experiment_id", "model", "dataset", "train"}
    missing = required.difference(spec)
    if missing:
        raise ValueError(f"Experiment spec missing keys: {sorted(missing)}")
    return spec


def validate_dataset_yaml(dataset_yaml: Path) -> dict[str, Any]:
    """Validate the train/validation contract and deliberately reject test usage."""
    with dataset_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must be a mapping: {dataset_yaml}")
    if "test" in data:
        raise ValueError("Milestone 2 training YAML must not contain a test split")
    for key in ("train", "val"):
        if key not in data:
            raise ValueError(f"Dataset YAML missing '{key}': {dataset_yaml}")
    root = Path(data.get("path", dataset_yaml.parent))
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    for key in ("train", "val"):
        split_path = Path(data[key])
        if not split_path.is_absolute():
            split_path = root / split_path
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset {key} path does not exist: {split_path}")
    return data


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = str(torch.__version__)
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["torch"] = None
        info["cuda_available"] = False
    try:
        import ultralytics

        info["ultralytics"] = str(ultralytics.__version__)
    except ImportError:
        info["ultralytics"] = None
    return info


def _extract_validation_metrics(result: Any) -> dict[str, float | None]:
    metrics = getattr(result, "results_dict", {}) or {}
    wanted = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "mAP50": "metrics/mAP50(B)",
        "mAP50-95": "metrics/mAP50-95(B)",
    }
    output: dict[str, float | None] = {}
    for name, key in wanted.items():
        value = metrics.get(key)
        output[name] = float(value) if value is not None else None
    return output


RESULT_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}


def summarize_results_csv(results_csv: Path) -> dict[str, Any]:
    """Summarize an Ultralytics ``results.csv`` at its best validation mAP50-95 epoch."""
    if not results_csv.is_file():
        return {"epochs_completed": 0, "best_epoch": None, "best_metrics": None}
    import pandas as pd

    frame = pd.read_csv(results_csv)
    frame.columns = [column.strip() for column in frame.columns]
    target = RESULT_COLUMNS["mAP50-95"]
    if frame.empty or target not in frame:
        return {"epochs_completed": int(len(frame)), "best_epoch": None, "best_metrics": None}
    best = frame.loc[frame[target].idxmax()]
    return {
        "epochs_completed": int(len(frame)),
        "best_epoch": int(best["epoch"]) if "epoch" in frame else None,
        "best_metrics": {
            name: float(best[column]) for name, column in RESULT_COLUMNS.items() if column in best
        },
    }


def card_path_for(spec: dict[str, Any], report_root: Path = REPORT_ROOT) -> Path:
    return report_root / "experiment_cards" / f"{spec['experiment_id']}.yaml"


def plan_run_action(save_dir: Path, card_path: Path) -> str:
    """Decide how to (re)start a run so interrupted GPU sessions continue safely.

    ``skip``   - an experiment card marks it completed and ``best.pt`` exists.
    ``resume`` - an interrupted run left ``weights/last.pt``.
    ``train``  - nothing usable exists yet.
    """
    if card_path.is_file() and (save_dir / "weights" / "best.pt").is_file():
        with card_path.open("r", encoding="utf-8") as handle:
            card = yaml.safe_load(handle) or {}
        if card.get("status") == "completed":
            return "skip"
    if (save_dir / "weights" / "last.pt").is_file():
        return "resume"
    return "train"


def write_experiment_card(
    spec: dict[str, Any],
    *,
    dataset_yaml: Path,
    save_dir: Path,
    validation_metrics: dict[str, float | None] | None = None,
    status: str = "completed",
    report_root: Path = REPORT_ROOT,
    extra: dict[str, Any] | None = None,
) -> Path:
    summary = summarize_results_csv(save_dir / "results.csv")
    card = {
        "experiment_id": spec["experiment_id"],
        "date": str(date.today()),
        "objective": spec.get("objective"),
        "git_commit": _git_commit(),
        "dataset_manifest": spec.get("dataset_manifest"),
        "dataset_yaml": str(dataset_yaml),
        "split_version": spec.get("split_version", "grazpedwri_seed42_patient_70_15_15"),
        "model": spec["model"],
        "pretrained_weights": spec.get("pretrained_weights", spec["model"]),
        "image_size": spec["train"].get("imgsz", 640),
        "preprocessing": spec.get("preprocessing"),
        "augmentations": spec.get("augmentations", {}),
        "hyperparameters": spec["train"],
        "hardware": _environment(),
        "seed": spec["train"].get("seed", 42),
        "checkpoint_selection_rule": "best validation mAP50-95(B)",
        "validation_metrics": validation_metrics,
        "epochs_completed": summary["epochs_completed"],
        "best_epoch": summary["best_epoch"],
        "best_epoch_metrics": summary["best_metrics"],
        "test_metrics": None,
        "artifacts": {
            "save_dir": str(save_dir),
            "best_checkpoint": str(save_dir / "weights" / "best.pt"),
            "results_csv": str(save_dir / "results.csv"),
        },
        "status": status,
        "notes": spec.get("notes"),
        "decision": spec.get("decision"),
    }
    if extra:
        card.update(extra)
    output = card_path_for(spec, report_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(json.loads(json.dumps(card, default=str)), handle, sort_keys=False)
    return output


def _validate_checkpoint(model: Any, save_dir: Path, dataset_yaml: Path) -> dict[str, float | None]:
    """Re-validate ``best.pt`` when a run is skipped or a resume returns no metrics."""
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        return {name: None for name in RESULT_COLUMNS}
    return _extract_validation_metrics(
        model.__class__(str(best)).val(data=str(dataset_yaml), split="val", plots=False)
    )


def run_experiment(
    spec_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    device: str | int | None = None,
    resume: bool = True,
    report_root: Path | None = None,
    extra_card_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one Ultralytics experiment and record a reproducible experiment card.

    With ``resume`` (default) a completed run is skipped and an interrupted run continues
    from ``weights/last.pt``; pass ``resume=False`` to always start a fresh run.
    """
    spec = load_experiment_spec(spec_path)
    report_root = report_root or (project_root / "reports")
    dataset_yaml = _resolve_path(spec["dataset"], project_root)
    validate_dataset_yaml(dataset_yaml)
    default_save_dir = report_root / "training_runs" / spec["experiment_id"]
    save_dir = _resolve_path(spec.get("save_dir", default_save_dir), project_root)
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    card_path = card_path_for(spec, report_root)
    action = plan_run_action(save_dir, card_path) if resume else "train"
    if action == "skip":
        with card_path.open("r", encoding="utf-8") as handle:
            card = yaml.safe_load(handle) or {}
        return {
            "experiment_id": spec["experiment_id"],
            "save_dir": str(save_dir),
            "card": str(card_path),
            "metrics": card.get("validation_metrics"),
            "action": "skip",
        }
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install the [ml] dependencies before training") from error

    write_experiment_card(
        spec,
        dataset_yaml=dataset_yaml,
        save_dir=save_dir,
        status="running",
        report_root=report_root,
        extra=extra_card_fields,
    )
    if action == "resume":
        model = YOLO(str(save_dir / "weights" / "last.pt"))
        train_args: dict[str, Any] = {"resume": True}
        if device is not None:
            train_args["device"] = device
    else:
        model = YOLO(str(spec["model"]))
        train_args = dict(spec["train"])
        train_args.update(
            {
                "data": str(dataset_yaml),
                "project": str(save_dir.parent),
                "name": save_dir.name,
                "exist_ok": True,
            }
        )
        if device is not None:
            train_args["device"] = device
    result = model.train(**train_args)
    metrics = _extract_validation_metrics(result)
    if all(value is None for value in metrics.values()):
        metrics = _validate_checkpoint(model, save_dir, dataset_yaml)
    card = write_experiment_card(
        spec,
        dataset_yaml=dataset_yaml,
        save_dir=save_dir,
        validation_metrics=metrics,
        report_root=report_root,
        extra=extra_card_fields,
    )
    return {
        "experiment_id": spec["experiment_id"],
        "save_dir": str(save_dir),
        "card": str(card),
        "metrics": metrics,
        "action": action,
    }


def save_json(data: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
