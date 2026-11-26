from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

FIGSHARE_API = "https://api.figshare.com/v2"


@dataclass(frozen=True)
class FigshareFile:
    file_id: int
    name: str
    size_bytes: int
    download_url: str
    supplied_md5: str | None


def get_article(article_id: int, timeout_seconds: int = 30) -> dict[str, Any]:
    response = requests.get(
        f"{FIGSHARE_API}/articles/{article_id}",
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Unexpected Figshare article response")
    return payload


def normalize_files(article: dict[str, Any]) -> list[FigshareFile]:
    files: list[FigshareFile] = []
    for item in article.get("files", []):
        files.append(
            FigshareFile(
                file_id=int(item["id"]),
                name=str(item["name"]),
                size_bytes=int(item.get("size", 0)),
                download_url=str(item["download_url"]),
                supplied_md5=item.get("supplied_md5") or item.get("computed_md5"),
            )
        )
    return files


def save_inventory(
    article: dict[str, Any],
    dataset_key: str,
    output_path: Path,
) -> dict[str, Any]:
    files = normalize_files(article)
    inventory = {
        "dataset_key": dataset_key,
        "article_id": article.get("id"),
        "title": article.get("title"),
        "doi": article.get("doi"),
        "version": article.get("version"),
        "published_date": article.get("published_date"),
        "modified_date": article.get("modified_date"),
        "license": article.get("license"),
        "url_public_api": article.get("url_public_api"),
        "url_private_api": article.get("url_private_api"),
        "total_size_bytes": sum(file.size_bytes for file in files),
        "files": [asdict(file) for file in files],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(inventory, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return inventory


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def md5_prefix(path: Path, byte_count: int, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    remaining = byte_count
    with path.open("rb") as handle:
        while remaining:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    if remaining:
        raise ValueError(f"File ended {remaining} bytes before expected prefix: {path}")
    return digest.hexdigest()


def load_inventory(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        inventory = json.load(handle)
    if not isinstance(inventory, dict) or not isinstance(inventory.get("files"), list):
        raise ValueError(f"Invalid inventory: {path}")
    return inventory


def select_inventory_files(
    inventory: dict[str, Any], requested_names: list[str]
) -> list[dict[str, Any]]:
    available = {str(item["name"]): item for item in inventory["files"]}
    missing = sorted(set(requested_names) - set(available))
    if missing:
        choices = ", ".join(sorted(available))
        raise KeyError(f"Unknown file(s): {', '.join(missing)}. Available: {choices}")
    return [available[name] for name in requested_names]


def download_figshare_file(
    file_record: dict[str, Any],
    destination_dir: Path,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    final_path = destination_dir / str(file_record["name"])
    partial_path = final_path.with_suffix(final_path.suffix + ".part")
    expected_size = int(file_record.get("size_bytes", 0))
    expected_md5 = file_record.get("supplied_md5")

    if final_path.exists():
        size_matches = not expected_size or final_path.stat().st_size == expected_size
        checksum_matches = not expected_md5 or md5_file(final_path) == expected_md5
        if size_matches and checksum_matches:
            return {"path": str(final_path), "status": "verified-existing"}
        raise ValueError(f"Existing file does not match inventory: {final_path}")

    if partial_path.exists() and expected_size:
        partial_size = partial_path.stat().st_size
        if partial_size >= expected_size:
            prefix_matches = (
                not expected_md5 or md5_prefix(partial_path, expected_size) == expected_md5
            )
            if not prefix_matches:
                raise ValueError(
                    f"Partial file reached expected size but checksum is invalid: {partial_path}"
                )
            if partial_size > expected_size:
                with partial_path.open("r+b") as handle:
                    handle.truncate(expected_size)
            partial_path.replace(final_path)
            return {"path": str(final_path), "status": "recovered-and-verified"}

    resume_at = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}
    with requests.get(
        str(file_record["download_url"]),
        headers=headers,
        stream=True,
        timeout=timeout_seconds,
    ) as response:
        response.raise_for_status()
        if resume_at and response.status_code != 206:
            resume_at = 0
        mode = "ab" if resume_at else "wb"
        total = expected_size if expected_size else None
        with (
            partial_path.open(mode) as handle,
            tqdm(
                total=total,
                initial=resume_at,
                unit="B",
                unit_scale=True,
                desc=final_path.name,
                disable=not sys.stderr.isatty(),
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))

    if expected_size and partial_path.stat().st_size != expected_size:
        raise ValueError(
            f"Downloaded size mismatch for {final_path.name}: "
            f"{partial_path.stat().st_size} != {expected_size}"
        )
    if expected_md5 and md5_file(partial_path) != expected_md5:
        raise ValueError(f"MD5 mismatch for {final_path.name}")
    partial_path.replace(final_path)
    return {"path": str(final_path), "status": "downloaded-and-verified"}
