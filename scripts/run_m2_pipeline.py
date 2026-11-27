"""Run Milestone 2 preprocessing and detector comparisons on a GPU host.

The script intentionally has no test-set path. It (re)builds the derived variants from the
frozen cohort and frozen labels, proves each variant matches the cohort before training,
selects preprocessing using validation mAP50-95, then trains the detector baselines on that
selected variant.

It is safe to re-run after an interrupted GPU session: completed experiments are skipped,
interrupted ones resume from ``weights/last.pt``, and the selection state in
``reports/m2_training_selection.json`` is updated after every experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from fractureguard.data.training_data import (
    HASH_CHECK_MODES,
    PREPROCESSING_VARIANTS,
    build_yolo_variant,
    verify_yolo_variant,
    write_dataset_yaml,
)
from fractureguard.paths import PROJECT_ROOT
from fractureguard.training import plan_run_action, run_experiment, summarize_results_csv

DETECTORS = {"yolo11n": "yolo11n.pt", "yolo26n": "yolo26n.pt", "yolo26s": "yolo26s.pt"}
EXPECTED_COUNTS = {"train": 7211, "val": 1517}
EXPECTED_BOXES = {"train": 3906, "val": 805}
AUGMENTATIONS = {
    "degrees": 7.0,
    "translate": 0.05,
    "scale": 0.10,
    "fliplr": 0.0,
    "flipud": 0.0,
    "mosaic": 0.2,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.15,
    "perspective": 0.0,
    "shear": 0.0,
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Raw GRAZPEDWRI-DX 16-bit source (needed only when a variant must be built)",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--cohort", type=Path, default=Path("data/manifests/grazpedwri_m2_cohort.csv")
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("data/manifests/grazpedwri_m2_labels.csv")
    )
    parser.add_argument(
        "--variants-root",
        "--output-root",
        dest="variants_root",
        type=Path,
        default=Path("data/derived/grazpedwri_yolo"),
        help="Where variants are built, or an existing (possibly read-only) prebuilt copy",
    )
    parser.add_argument("--hash-check", choices=HASH_CHECK_MODES, default="strict")
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0, 0,1 or cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--preprocess-epochs", type=int, default=25)
    parser.add_argument("--final-epochs", type=int, default=100)
    parser.add_argument("--stage", choices=("all", "preprocess", "detectors"), default="all")
    parser.add_argument("--variants", nargs="+", choices=PREPROCESSING_VARIANTS)
    parser.add_argument("--models", nargs="+", choices=sorted(DETECTORS))
    parser.add_argument(
        "--preprocessing",
        choices=PREPROCESSING_VARIANTS,
        help="Override the recorded preprocessing selection (logged as a manual override)",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        help="Do not start a new experiment after this many hours (keeps a Kaggle session "
        "inside its limit; completed work is kept and the next session continues)",
    )
    parser.add_argument("--skip-decode-check", action="store_true")
    parser.add_argument(
        "--fresh", action="store_true", help="Do not skip completed or resume interrupted runs"
    )
    parser.add_argument("--allow-smaller-cohort", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _absolute(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec(
    *,
    experiment_id: str,
    model: str,
    dataset_yaml: Path,
    objective: str,
    epochs: int,
    batch: int,
    workers: int,
    save_dir: Path,
    preprocessing: str,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "objective": objective,
        "dataset": str(dataset_yaml),
        "dataset_manifest": "data/manifests/grazpedwri_m2_cohort.csv",
        "model": model,
        "pretrained_weights": model,
        "preprocessing": preprocessing,
        "augmentations": dict(AUGMENTATIONS),
        "train": {
            "imgsz": 640,
            "epochs": epochs,
            "batch": batch,
            "patience": 7 if epochs <= 25 else 20,
            "seed": 42,
            "deterministic": True,
            "workers": workers,
            "pretrained": True,
            "plots": True,
            "save_period": 5,
            **AUGMENTATIONS,
        },
        "save_dir": str(save_dir),
        "notes": "Validation-only selection; the frozen internal test split is excluded.",
    }


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = args.project_root.resolve()
        self.cohort = _absolute(args.cohort, self.root)
        self.labels = _absolute(args.labels, self.root)
        self.variants_root = _absolute(args.variants_root, self.root)
        self.dataset_root = args.dataset_root.resolve() if args.dataset_root else None
        self.report_root = self.root / "reports"
        self.run_root = self.report_root / "training_runs"
        self.work_root = self.root / "work" / "m2"
        self.state_path = self.report_root / "m2_training_selection.json"
        self.state = self._load_state()
        self.verified: dict[str, Path] = {}
        self.started = time.monotonic()
        self.out_of_time = False

    def _budget_left(self) -> bool:
        if self.args.max_hours is None:
            return True
        elapsed = (time.monotonic() - self.started) / 3600
        if elapsed >= self.args.max_hours:
            self.out_of_time = True
            print(f"[budget] {elapsed:.2f} h used; not starting another experiment", flush=True)
            return False
        return True

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"preprocessing": {}, "detectors": {}}

    def _save_state(self) -> None:
        self.state["updated_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        self.state["frozen_inputs"] = {
            "cohort_sha256": _sha256(self.cohort),
            "labels_sha256": _sha256(self.labels),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

    def prepared_variant(self, variant: str) -> Path:
        """Build if necessary, verify against the frozen cohort, return a host dataset YAML."""
        if variant in self.verified:
            return self.verified[variant]
        variant_root = self.variants_root / variant
        built = (variant_root / "build_metadata.json").is_file() and (
            variant_root / "images" / "val"
        ).is_dir()
        if not built:
            if self.dataset_root is None:
                raise SystemExit(
                    f"Variant {variant} is not built under {self.variants_root}; "
                    "pass --dataset-root pointing at the raw GRAZPEDWRI-DX images."
                )
            print(f"[build] {variant} from {self.dataset_root}", flush=True)
            build_yolo_variant(
                self.cohort,
                self.dataset_root,
                self.variants_root,
                variant,
                workers=self.args.workers,
                labels_csv=self.labels,
                hash_check=self.args.hash_check,
            )
        print(f"[verify] {variant}", flush=True)
        report = verify_yolo_variant(
            variant_root,
            self.cohort,
            self.labels,
            workers=self.args.workers,
            decode_images=not self.args.skip_decode_check,
            report_path=self.work_root / "verification" / f"{variant}.json",
        )
        counts = {split: values["images"] for split, values in report["counts"].items()}
        boxes = {split: values["boxes"] for split, values in report["counts"].items()}
        if not self.args.allow_smaller_cohort and (
            counts != EXPECTED_COUNTS or boxes != EXPECTED_BOXES
        ):
            raise SystemExit(
                f"{variant} does not match the frozen protocol: images={counts} "
                f"(expected {EXPECTED_COUNTS}), boxes={boxes} (expected {EXPECTED_BOXES})"
            )
        dataset_yaml = write_dataset_yaml(
            variant_root, self.work_root / "datasets" / f"{variant}.yaml"
        )
        self.state.setdefault("verification", {})[variant] = {
            "counts": report["counts"],
            "images_decoded": report["images_decoded"],
            "passed": report["passed"],
        }
        self.verified[variant] = dataset_yaml
        return dataset_yaml

    def run(
        self,
        *,
        experiment_id: str,
        model: str,
        variant: str,
        epochs: int,
        objective: str,
        save_name: str,
    ) -> dict[str, Any]:
        save_dir = self.run_root / save_name
        card = self.report_root / "experiment_cards" / f"{experiment_id}.yaml"
        if not self.args.fresh and plan_run_action(save_dir, card) == "skip":
            # Completed in an earlier session: do not rebuild or re-verify its variant.
            print(f"[skip] {experiment_id} already completed", flush=True)
            summary = summarize_results_csv(save_dir / "results.csv")
            stored = yaml.safe_load(card.read_text(encoding="utf-8")) or {}
            return {
                "experiment_id": experiment_id,
                "save_dir": str(save_dir),
                "card": str(card),
                "metrics": stored.get("validation_metrics"),
                "action": "skip",
                "epochs_completed": summary["epochs_completed"],
                "best_epoch": summary["best_epoch"],
                "best_mAP50-95": (summary["best_metrics"] or {}).get("mAP50-95"),
            }
        dataset_yaml = self.prepared_variant(variant)
        spec = _spec(
            experiment_id=experiment_id,
            model=model,
            dataset_yaml=dataset_yaml,
            objective=objective,
            epochs=epochs,
            batch=self.args.batch,
            workers=self.args.workers,
            save_dir=save_dir,
            preprocessing=variant,
        )
        spec_path = self.work_root / "specs" / f"{experiment_id}.yaml"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        print(f"[run] {experiment_id}", flush=True)
        result = run_experiment(
            spec_path,
            project_root=self.root,
            device=self.args.device,
            resume=not self.args.fresh,
            report_root=self.report_root,
            extra_card_fields={
                "data_verification": self.state.get("verification", {}).get(variant),
                "frozen_labels_sha256": _sha256(self.labels),
                "cohort_sha256": _sha256(self.cohort),
            },
        )
        summary = summarize_results_csv(Path(result["save_dir"]) / "results.csv")
        best = (summary["best_metrics"] or {}).get("mAP50-95")
        return {
            **result,
            "epochs_completed": summary["epochs_completed"],
            "best_epoch": summary["best_epoch"],
            "best_mAP50-95": best,
        }

    def preprocessing_screen(self) -> None:
        for variant in self.args.variants or PREPROCESSING_VARIANTS:
            if not self._budget_left():
                return
            record = self.run(
                experiment_id=f"20261116_preprocess_yolo26n_640_{variant}_seed42",
                model=DETECTORS["yolo26n"],
                variant=variant,
                epochs=self.args.preprocess_epochs,
                objective="Select 16-bit radiograph preprocessing using clean validation mAP50-95.",
                save_name=f"preprocess_yolo26n_640_{variant}_seed42",
            )
            self.state["preprocessing"][variant] = record
            self._save_state()
        self._select_preprocessing()

    def _select_preprocessing(self) -> None:
        scored = {
            name: record["best_mAP50-95"]
            for name, record in self.state["preprocessing"].items()
            if record.get("best_mAP50-95") is not None
        }
        if set(scored) != set(PREPROCESSING_VARIANTS):
            print(f"[select] waiting for all variants; have {sorted(scored)}", flush=True)
            return
        if max(scored.values()) <= 0.0:
            raise SystemExit(
                "Every preprocessing run has mAP50-95 <= 0; this indicates a data or label "
                "problem, not a model result. Inspect the verification reports before continuing."
            )
        selected = max(scored, key=scored.__getitem__)
        self.state["selected_preprocessing"] = selected
        self.state["selection_rule"] = "max best-epoch validation mAP50-95(B), YOLO26n, 640"
        self._save_state()
        print(f"[select] preprocessing = {selected} {scored}", flush=True)

    def detector_comparison(self) -> None:
        variant = self.args.preprocessing or self.state.get("selected_preprocessing")
        if variant is None:
            raise SystemExit("No preprocessing selected yet; run the preprocess stage first.")
        if self.args.preprocessing and self.args.preprocessing != self.state.get(
            "selected_preprocessing"
        ):
            self.state["preprocessing_override"] = self.args.preprocessing
        for model_id in self.args.models or list(DETECTORS):
            if not self._budget_left():
                return
            record = self.run(
                experiment_id=f"20261116_detector_{model_id}_640_{variant}_seed42",
                model=DETECTORS[model_id],
                variant=variant,
                epochs=self.args.final_epochs,
                objective="Compare single-view detector baselines on the selected preprocessing.",
                save_name=f"detector_{model_id}_640_{variant}_seed42",
            )
            self.state["detectors"][model_id] = record
            self._save_state()
        scored = {
            name: record["best_mAP50-95"]
            for name, record in self.state["detectors"].items()
            if record.get("best_mAP50-95") is not None
        }
        if set(scored) == set(DETECTORS):
            self.state["best_validation_detector"] = max(scored, key=scored.__getitem__)
            self._save_state()


def main() -> int:
    args = _args()
    pipeline = Pipeline(args)
    if args.stage in ("all", "preprocess"):
        pipeline.preprocessing_screen()
    if args.stage in ("all", "detectors") and not pipeline.out_of_time:
        if args.stage == "all" and "selected_preprocessing" not in pipeline.state:
            print("[stop] preprocessing screen incomplete; re-run to continue", flush=True)
        else:
            pipeline.detector_comparison()
    print(json.dumps(pipeline.state, indent=2))
    print(f"Selection report: {pipeline.state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
