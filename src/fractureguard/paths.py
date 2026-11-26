from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"
DATA_ROOT = PROJECT_ROOT / "data"
MANIFEST_ROOT = DATA_ROOT / "manifests"
REPORT_ROOT = PROJECT_ROOT / "reports"


def ensure_project_directories() -> None:
    """Create generated-data directories without touching raw dataset contents."""
    for path in (
        DATA_ROOT / "raw",
        DATA_ROOT / "derived",
        DATA_ROOT / "cache",
        MANIFEST_ROOT,
        MANIFEST_ROOT / "inventory",
        DATA_ROOT / "splits",
        REPORT_ROOT / "figures",
        REPORT_ROOT / "tables",
        REPORT_ROOT / "predictions",
        REPORT_ROOT / "experiment_cards",
    ):
        path.mkdir(parents=True, exist_ok=True)
