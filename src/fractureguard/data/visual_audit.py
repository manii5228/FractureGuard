from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from fractureguard.data.training_data import PREPROCESSING_VARIANTS


def stratified_overlay_sample(
    frame: pd.DataFrame,
    count: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    strata = ["split", "projection", "fracture_visible"]
    per_group = max(1, math.ceil(count / frame.groupby(strata).ngroups))
    sampled = pd.concat(
        [
            group.sample(n=min(per_group, len(group)), random_state=seed)
            for _, group in frame.groupby(strata, sort=True)
        ],
        ignore_index=True,
    ).drop_duplicates("image_id")
    if len(sampled) > count:
        sampled = sampled.sample(n=count, random_state=seed)
    elif len(sampled) < count:
        remaining = frame.loc[~frame["image_id"].isin(sampled["image_id"])]
        fill = remaining.sample(n=count - len(sampled), random_state=seed)
        sampled = pd.concat([sampled, fill], ignore_index=True)
    return sampled.sort_values([*strata, "image_id"], kind="stable")


def _fracture_boxes(annotation_path: Path) -> list[tuple[int, int, int, int]]:
    root = ET.parse(annotation_path).getroot()
    boxes: list[tuple[int, int, int, int]] = []
    for item in root.findall("./object"):
        if (item.findtext("name") or "").strip().lower() != "fracture":
            continue
        box = item.find("bndbox")
        if box is None:
            continue
        boxes.append(
            (
                int(box.findtext("xmin", "0")),
                int(box.findtext("ymin", "0")),
                int(box.findtext("xmax", "0")),
                int(box.findtext("ymax", "0")),
            )
        )
    return boxes


def _render_case(
    row: pd.Series,
    dataset_root: Path,
    tile_size: tuple[int, int],
) -> Image.Image:
    tile_width, tile_height = tile_size
    header_height = 32
    image_path = dataset_root / str(row["image_path"])
    annotation_path = dataset_root / str(row["label_path"])
    with Image.open(image_path) as source:
        original_width, original_height = source.size
        pixels = np.asarray(source, dtype=np.float32)
        lower, upper = np.percentile(pixels, (1.0, 99.0))
        if upper <= lower:
            upper = lower + 1.0
        normalized = np.clip((pixels - lower) * (255.0 / (upper - lower)), 0, 255)
        display = Image.fromarray(normalized.astype(np.uint8), mode="L").convert("RGB")
        display.thumbnail((tile_width, tile_height - header_height), Image.Resampling.LANCZOS)

    tile = Image.new("RGB", tile_size, "black")
    offset_x = (tile_width - display.width) // 2
    offset_y = header_height + (tile_height - header_height - display.height) // 2
    tile.paste(display, (offset_x, offset_y))
    draw = ImageDraw.Draw(tile)
    scale_x = display.width / original_width
    scale_y = display.height / original_height
    for xmin, ymin, xmax, ymax in _fracture_boxes(annotation_path):
        draw.rectangle(
            (
                offset_x + xmin * scale_x,
                offset_y + ymin * scale_y,
                offset_x + xmax * scale_x,
                offset_y + ymax * scale_y,
            ),
            outline=(255, 50, 50),
            width=3,
        )
    status = "+" if bool(row["fracture_visible"]) else "-"
    label = f"{row['image_id']} | {row['split']} | {row['projection']} | Fx {status}"
    draw.text((5, 8), label, fill="white", font=ImageFont.load_default())
    return tile


def create_graz_overlay_sheets(
    ready_manifest_csv: Path,
    dataset_root: Path,
    output_dir: Path,
    count: int = 100,
    cases_per_sheet: int = 25,
    seed: int = 42,
) -> list[Path]:
    frame = pd.read_csv(ready_manifest_csv)
    sampled = stratified_overlay_sample(frame, count=count, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_size = (320, 400)
    columns = 5
    rows = math.ceil(cases_per_sheet / columns)
    outputs: list[Path] = []
    for sheet_index, start in enumerate(range(0, len(sampled), cases_per_sheet), start=1):
        subset = sampled.iloc[start : start + cases_per_sheet]
        sheet = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "#202020")
        for tile_index, (_, case) in enumerate(subset.iterrows()):
            tile = _render_case(case, dataset_root, tile_size)
            x = (tile_index % columns) * tile_size[0]
            y = (tile_index // columns) * tile_size[1]
            sheet.paste(tile, (x, y))
        output = output_dir / f"graz_overlay_audit_{sheet_index:02d}.jpg"
        sheet.save(output, quality=90)
        outputs.append(output)
    sampled.to_csv(output_dir / "graz_overlay_audit_sample.csv", index=False)
    return outputs


def _render_yolo_case(
    image_path: Path,
    label_path: Path,
    title: str,
    tile_size: tuple[int, int] = (260, 340),
) -> Image.Image:
    header_height = 30
    with Image.open(image_path) as source:
        display = source.convert("RGB")
        original_width, original_height = display.size
        display.thumbnail((tile_size[0], tile_size[1] - header_height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", tile_size, "black")
    offset_x = (tile_size[0] - display.width) // 2
    offset_y = header_height + (tile_size[1] - header_height - display.height) // 2
    tile.paste(display, (offset_x, offset_y))
    draw = ImageDraw.Draw(tile)
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            _, center_x, center_y, width, height = (float(value) for value in line.split())
            xmin = (center_x - width / 2) * display.width + offset_x
            xmax = (center_x + width / 2) * display.width + offset_x
            ymin = (center_y - height / 2) * display.height + offset_y
            ymax = (center_y + height / 2) * display.height + offset_y
            draw.rectangle((xmin, ymin, xmax, ymax), outline=(255, 50, 50), width=3)
    draw.text((4, 8), title, fill="white", font=ImageFont.load_default())
    return tile


def create_yolo_variant_comparison(
    cohort_csv: Path,
    variants_root: Path,
    output_path: Path,
    cases_per_class: int = 4,
    seed: int = 42,
) -> Path:
    cohort = pd.read_csv(cohort_csv)
    positives = cohort.loc[
        cohort["fracture_visible"].astype(str).str.lower().eq("true") & cohort["m2_split"].eq("val")
    ].sample(n=cases_per_class, random_state=seed)
    negatives = cohort.loc[
        ~cohort["fracture_visible"].astype(str).str.lower().eq("true")
        & cohort["m2_split"].eq("val")
    ].sample(n=cases_per_class, random_state=seed)
    cases = pd.concat([negatives, positives], ignore_index=True)
    tile_size = (260, 340)
    sheet = Image.new(
        "RGB",
        (len(cases) * tile_size[0], len(PREPROCESSING_VARIANTS) * tile_size[1]),
        "#202020",
    )
    for row_index, variant in enumerate(PREPROCESSING_VARIANTS):
        for column_index, case in cases.iterrows():
            image_id = str(case["image_id"])
            split = str(case["m2_split"])
            root = variants_root / variant
            tile = _render_yolo_case(
                root / "images" / split / f"{image_id}.png",
                root / "labels" / split / f"{image_id}.txt",
                f"{variant} | {image_id}",
                tile_size,
            )
            sheet.paste(tile, (column_index * tile_size[0], row_index * tile_size[1]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return output_path
