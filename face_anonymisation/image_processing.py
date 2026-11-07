"""Orientation-safe image I/O, bounded face expansion, and Gaussian blur."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    from .detectors import Detection
except ImportError:  # pragma: no cover
    from detectors import Detection


@dataclass
class ImageAsset:
    """Decoded pixels plus metadata needed to save the same image format."""

    image: Image.Image
    image_format: str
    metadata: Dict[str, Any]


def load_image(path: Path) -> ImageAsset:
    """Load pixels without applying EXIF orientation, preserving raw coordinates."""
    with Image.open(str(path)) as source:
        source.load()
        image_format = source.format
        if image_format is None:
            raise ValueError("Image format could not be determined")
        has_alpha = "A" in source.getbands() or "transparency" in source.info
        pixels = source.convert("RGBA" if has_alpha else "RGB")
        metadata = {
            "exif": source.info.get("exif"),
            "icc_profile": source.info.get("icc_profile"),
            "dpi": source.info.get("dpi"),
        }
    return ImageAsset(pixels, image_format, metadata)


def expand_box(
    box: Tuple[float, float, float, float],
    padding: float,
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Expand xyxy by padding on each side and clip to ``(width, height)``."""
    if padding < 0:
        raise ValueError("Padding cannot be negative")
    width, height = image_size
    x1, y1, x2, y2 = box
    box_width, box_height = max(0.0, x2 - x1), max(0.0, y2 - y1)
    left = max(0, min(width, int(np.floor(x1 - box_width * padding))))
    top = max(0, min(height, int(np.floor(y1 - box_height * padding))))
    right = max(0, min(width, int(np.ceil(x2 + box_width * padding))))
    bottom = max(0, min(height, int(np.ceil(y2 + box_height * padding))))
    return left, top, right, bottom


def _odd_kernel(length: int, strength: float) -> int:
    """Choose an odd Gaussian kernel based on a region edge and relative strength."""
    target = max(3, int(round(max(1, length) * strength * 2.0)))
    return target if target % 2 else target + 1


def gaussian_kernel_size(region_size: Tuple[int, int], strength: float) -> Tuple[int, int]:
    """Return valid odd ``(width, height)`` kernels for an image region."""
    if not 0.0 < strength <= 1.0:
        raise ValueError("Blur strength must be greater than 0 and at most 1")
    width, height = region_size
    return _odd_kernel(width, strength), _odd_kernel(height, strength)


def blur_detections(
    image: Image.Image,
    detections: Iterable[Detection],
    padding: float = 0.30,
    blur_strength: float = 0.5,
) -> Tuple[Image.Image, list]:
    """Blur detected face regions after expanding their boxes."""
    if not 0.0 < blur_strength <= 1.0:
        raise ValueError("Blur strength must be greater than 0 and at most 1")
    if padding < 0:
        raise ValueError("Padding cannot be negative")

    width, height = image.size
    output = image.copy()
    regions = []
    try:
        import cv2
    except ImportError:
        cv2 = None

    # Blur each region from the original so results do not depend on detector order.
    for detection in detections:
        region = expand_box(detection.box, padding, image.size)
        left, top, right, bottom = region
        if right <= left or bottom <= top:
            continue
        roi = image.crop(region)
        kernel = gaussian_kernel_size((right - left, bottom - top), blur_strength)
        if cv2 is not None:
            pixels = np.asarray(roi).copy()
            blurred = cv2.GaussianBlur(pixels, kernel, sigmaX=0, sigmaY=0)
            blurred_image = Image.fromarray(blurred)
        else:
            # Pillow fallback for environments without OpenCV.
            radius = max(0.5, (max(kernel) - 1) / 6.0)
            blurred_image = roi.filter(ImageFilter.GaussianBlur(radius=radius))
        output.paste(blurred_image, (left, top))
        regions.append(region)
    return output, regions


def save_image(
    asset: ImageAsset,
    destination: Path,
    overwrite: bool = False,
    jpeg_quality: int = 100,
) -> None:
    """Atomically encode a transformed image in its original file format."""
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError("Output already exists: {}".format(destination))
    suffix = destination.suffix or ".img"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(destination.stem), suffix=suffix, dir=str(destination.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    options = {key: value for key, value in asset.metadata.items() if value is not None}
    if asset.image_format.upper() in ("JPEG", "JPG"):
        options.update({"quality": jpeg_quality, "subsampling": 0})
    try:
        asset.image.save(str(temporary), format=asset.image_format, **options)
        if destination.exists() and not overwrite:
            raise FileExistsError("Output already exists: {}".format(destination))
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()


def copy_image_bytes(source: Path, destination: Path, overwrite: bool = False) -> None:
    """Copy a no-detection image byte-for-byte, atomically and with metadata."""
    import shutil

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError("Output already exists: {}".format(destination))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(destination.stem), suffix=destination.suffix,
        dir=str(destination.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(str(source), str(temporary))
        shutil.copystat(str(source), str(temporary))
        if destination.exists() and not overwrite:
            raise FileExistsError("Output already exists: {}".format(destination))
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()


def save_review_image(
    image: Image.Image,
    detections: Iterable[Detection],
    destination: Path,
    padding: float,
    overwrite: bool = False,
) -> None:
    """Save a JPEG review image with boxes and model scores over blurred pixels."""
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    review = image.convert("RGB")
    draw = ImageDraw.Draw(review)
    for detection in detections:
        left, top, right, bottom = expand_box(detection.box, padding, image.size)
        draw.rectangle((left, top, max(left, right - 1), max(top, bottom - 1)), outline=(0, 255, 80), width=2)
        labels = []
        for source in detection.sources:
            labels.append("{} {:.2f}".format(source, detection.scores[source]))
        label = ", ".join(labels)
        draw.text((left, max(0, top - 14)), label, fill=(0, 255, 80))
    destination.parent.mkdir(parents=True, exist_ok=True)
    review.save(str(destination), format="JPEG", quality=90)
