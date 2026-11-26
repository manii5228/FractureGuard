from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fractureguard.paths import CONFIG_ROOT


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def load_dataset_config(dataset_key: str) -> dict[str, Any]:
    config = load_yaml(CONFIG_ROOT / "datasets.yaml")
    datasets = config.get("datasets", {})
    if dataset_key not in datasets:
        choices = ", ".join(sorted(datasets))
        raise KeyError(f"Unknown dataset '{dataset_key}'. Available: {choices}")
    dataset = datasets[dataset_key]
    if not isinstance(dataset, dict):
        raise ValueError(f"Invalid configuration for dataset '{dataset_key}'")
    return dataset
