"""Run BLPR on sample images and save annotated predictions."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys

import cv2

from blpr_pipeline import (
    CHAR_MODEL,
    INFERENCE_DIR,
    INFERENCE_PICTURES_DIR,
    LLMPlateReader,
    PLATE_MODEL,
    PICTURES_DIR,
    recognize_plate,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_COLOR = (40, 220, 80)


@dataclass(frozen=True)
class PlateDetection:
    box: tuple[int, int, int, int]
    text: str


def infer_image(image, plate_model, char_model, llm_reader=None):
    """Detect and read every plate without consulting ground-truth labels."""
    height, width = image.shape[:2]
    detections = []
    for result in plate_model.predict(image, verbose=False, conf=0.5):
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            if str(plate_model.names[class_id]).lower() not in {"plate", "placa"}:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
            x2, y2 = max(0, min(x2, width)), max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue

            roi = image[
                max(0, y1 - 10):min(height, y2 + 10),
                max(0, x1 - 10):min(width, x2 + 10),
            ]
            text = recognize_plate(roi, char_model, llm_reader)
            detections.append(PlateDetection((x1, y1, x2, y2), text))
    return detections


def draw_text_box(image, text, x, y, scale):
    """Draw a readable label that stays within the image boundaries."""
    height, width = image.shape[:2]
    padding = max(2, round(4 * scale))
    thickness = max(1, round(scale))
    (text_width, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    available_width = max(1, width - 2 * padding - 1)
    if text_width > available_width:
        scale *= available_width / text_width
        thickness = max(1, round(scale))
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    box_width = min(width, text_width + 2 * padding)
    box_height = min(height, text_height + baseline + 2 * padding)
    x = max(0, min(x, width - box_width))
    y = max(0, min(y, height - box_height))
    cv2.rectangle(image, (x, y), (x + box_width - 1, y + box_height - 1), (24, 24, 24), -1)
    cv2.putText(
        image, text, (x + padding, y + padding + text_height),
        FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )
    return box_height


def annotate_image(image, detections):
    """Add numbered plate boxes, per-box text, and a top-left summary."""
    annotated = image.copy()
    height, width = image.shape[:2]
    scale = max(0.45, min(height, width) / 900)
    thickness = max(2, round(scale * 2))
    labels = []
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection.box
        label = f"{index}: {detection.text or 'Unreadable'}"
        labels.append(label)
        cv2.rectangle(
            annotated, (x1, y1), (min(x2, width - 1), min(y2, height - 1)),
            BOX_COLOR, thickness,
        )
        (_, text_height), baseline = cv2.getTextSize(label, FONT, scale, max(1, round(scale)))
        label_height = text_height + baseline + 2 * max(2, round(4 * scale))
        label_y = y1 - label_height if y1 >= label_height else y1 + thickness
        draw_text_box(annotated, label, x1, label_y, scale)

    summary = ["Detected plates", *labels] if labels else ["No plates detected"]
    margin = max(2, round(6 * scale))
    (_, line_height), baseline = cv2.getTextSize("Ag", FONT, scale, max(1, round(scale)))
    required_height = len(summary) * (line_height + baseline + 2 * max(2, round(4 * scale)))
    summary_scale = scale * min(1, max(1, height - 2 * margin) / required_height)
    y = margin
    for line in summary:
        y += draw_text_box(annotated, line, margin, y, summary_scale)
    return annotated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PICTURES_DIR,
        help="Input image folder (default: assets/pictures/; labels.csv is not required).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=INFERENCE_PICTURES_DIR,
        help="Annotated image folder, created if missing (default: assets/inference/pictures/).",
    )
    parser.add_argument("--no-vlm", action="store_true", help="Run OCR without the Gemma fallback.")
    args = parser.parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_dir.is_dir():
        parser.error(f"Input folder not found: {data_dir}")
    if data_dir == output_dir:
        parser.error("Input and output folders must differ to preserve the sample images.")

    image_paths = sorted(
        path for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        parser.error(f"No supported images found in {data_dir}")
    for model_path in (PLATE_MODEL, CHAR_MODEL):
        if not model_path.is_file():
            parser.error(f"Model checkpoint not found: {model_path}")

    default_output_dir = INFERENCE_PICTURES_DIR.resolve()
    if output_dir == default_output_dir:
        if INFERENCE_DIR.is_symlink() or INFERENCE_PICTURES_DIR.is_symlink():
            parser.error(f"Refusing to clear inference output symlink: {INFERENCE_DIR}")
        if INFERENCE_PICTURES_DIR.exists():
            if not INFERENCE_PICTURES_DIR.is_dir():
                parser.error(f"Inference output path is not a folder: {INFERENCE_PICTURES_DIR}")
            try:
                shutil.rmtree(INFERENCE_PICTURES_DIR)
            except OSError as error:
                parser.error(f"Cannot clear inference folder {INFERENCE_PICTURES_DIR}: {error}")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        parser.error(f"Cannot create output folder {output_dir}: {error}")

    from ultralytics import YOLO

    print("Loading YOLO models...")
    plate_model = YOLO(str(PLATE_MODEL), task="detect")
    char_model = YOLO(str(CHAR_MODEL), task="detect")
    llm_reader = None if args.no_vlm else LLMPlateReader()

    saved = 0
    failures = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}", file=sys.stderr)
            failures += 1
            continue
        detections = infer_image(image, plate_model, char_model, llm_reader)
        annotated = annotate_image(image, detections)
        output_path = output_dir / image_path.name
        try:
            written = cv2.imwrite(str(output_path), annotated)
        except cv2.error as error:
            print(f"Could not write {output_path}: {error}", file=sys.stderr)
            failures += 1
            continue
        if not written:
            print(f"Could not write {output_path}", file=sys.stderr)
            failures += 1
            continue
        saved += 1
        readings = ", ".join(item.text or "Unreadable" for item in detections)
        print(f"{image_path.name}: {readings or 'No plates detected'} -> {output_path}")

    print(f"Saved {saved}/{len(image_paths)} annotated images to {output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
