"""Package the code and frozen Milestone 2 manifests for upload as a Kaggle dataset.

The bundle is small (code + cohort + frozen labels, no images). On Kaggle the images are
rebuilt from the attached GRAZPEDWRI-DX source and checked against the audited SHA-256
hashes, so the GPU run uses exactly the locally audited cohort.

    python scripts/build_kaggle_bundle.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TRAINING_INPUTS = (
    "data/manifests/grazpedwri_m2_cohort.csv",
    "data/manifests/grazpedwri_m2_labels.csv",
)
INCLUDE = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "FRACTUREGUARD_PROJECT_PLAN.md",
    "configs",
    "scripts",
    "src/fractureguard",
    "tests",
    "notebooks/kaggle_m2_training.ipynb",
    *REQUIRED_TRAINING_INPUTS,
    "data/manifests/DATA_AUDIT_REPORT.md",
)
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def _files(root: Path) -> list[Path]:
    found: list[Path] = []
    for entry in INCLUDE:
        path = root / entry
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(
                p
                for p in sorted(path.rglob("*"))
                if p.is_file() and not SKIP_PARTS.intersection(p.parts) and p.suffix != ".pyc"
            )
    return found


def build_bundle(root: Path, output: Path) -> dict[str, object]:
    missing = [name for name in REQUIRED_TRAINING_INPUTS if not (root / name).is_file()]
    if missing:
        required = ", ".join(missing)
        raise FileNotFoundError(
            f"Frozen training inputs are missing ({required}). See data/README.md."
        )

    files = _files(root)
    manifest = {
        "bundle": "fractureguard_m2",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "work" / "kaggle" / "fractureguard_m2_bundle.zip",
    )
    args = parser.parse_args()
    manifest = build_bundle(args.project_root.resolve(), args.output)
    print(f"Bundle: {args.output} ({len(manifest['files'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
