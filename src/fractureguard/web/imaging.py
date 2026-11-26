from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from django.conf import settings
from PIL import Image

from .security import digest

VIEWS = ("frontal", "lateral")
VIEW_LABELS = {"frontal": "AP/PA", "lateral": "Lateral"}


@dataclass(frozen=True)
class PreparedImage:
    png: bytes
    preview: bytes
    width: int
    height: int
    fingerprint: str


def normalized(pixels: np.ndarray) -> np.ndarray:
    values = pixels.astype(np.float32)
    low, high = np.percentile(values, (1.0, 99.0))
    if high <= low:
        high = low + 1.0
    return np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def prepare_upload(upload) -> PreparedImage:
    if not upload:
        raise ValueError("Choose a PNG image.")
    if upload.size > settings.MAX_UPLOAD_BYTES:
        raise ValueError("Choose a PNG file no larger than 10 MiB.")
    if not upload.name.lower().endswith(".png"):
        raise ValueError("Only PNG images are accepted.")
    raw = upload.read(settings.MAX_UPLOAD_BYTES + 1)
    if not raw.startswith(b"\x89PNG\r\n\x1a\n") or len(raw) > settings.MAX_UPLOAD_BYTES:
        raise ValueError("The file is not a valid PNG within the size limit.")
    try:
        with Image.open(BytesIO(raw), formats=["PNG"]) as original:
            width, height = original.size
            if max(width, height) > 4096 or width * height > 8_000_000:
                raise ValueError("Use an image within 4,096 pixels per side and 8 megapixels.")
            if original.mode not in {"L", "I;16", "I;16B", "I;16L"}:
                raise ValueError("Upload an 8-bit or 16-bit grayscale radiograph.")
            if getattr(original, "n_frames", 1) != 1:
                raise ValueError("Upload a single radiograph per view.")
            original.verify()
        with Image.open(BytesIO(raw), formats=["PNG"]) as loaded:
            pixels = np.asarray(loaded)
            if pixels.min() == pixels.max():
                raise ValueError("The image has no visible grayscale detail.")
            clean = Image.fromarray(pixels.astype(np.uint8 if loaded.mode == "L" else np.uint16))
            canonical_png = png_bytes(clean)
            preview = png_bytes(Image.fromarray(normalized(pixels)))
            fingerprint = digest(f"{width}:{height}:".encode() + pixels.astype("<u2").tobytes())
            return PreparedImage(canonical_png, preview, width, height, fingerprint)
    except (OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValueError("The PNG could not be decoded. Export the radiograph again.") from error
