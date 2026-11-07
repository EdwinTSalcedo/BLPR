#!/usr/bin/env python3
"""Recursively detect and Gaussian-blur faces in the BLPR-A image dataset."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from tqdm import tqdm

try:
    from .detectors import Detection, create_detectors, merge_detections
    from .image_processing import (
        ImageAsset,
        blur_detections,
        copy_image_bytes,
        load_image,
        save_image,
        save_review_image,
    )
except ImportError:  # pragma: no cover
    from detectors import Detection, create_detectors, merge_detections
    from image_processing import (
        ImageAsset,
        blur_detections,
        copy_image_bytes,
        load_image,
        save_image,
        save_review_image,
    )


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CSV_FIELDS = [
    "relative_image_path",
    "processing_status",
    "yunet_detections",
    "scrfd_detections",
    "final_merged_detections",
    "yunet_confidence_scores",
    "scrfd_confidence_scores",
    "final_bounding_boxes_xyxy",
    "detectors_per_final_detection",
    "detections_excluded_for_protected_label_overlap",
    "processing_time_seconds",
    "error_message",
]


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_paths(input_dir: Path, output_dir: Path, review_dir: Optional[Path]) -> None:
    if not input_dir.is_dir():
        raise ValueError("Input directory does not exist or is not a directory: {}".format(input_dir))
    source = input_dir.resolve()
    output = output_dir.resolve()
    if output == source or _is_within(output, source):
        raise ValueError("Output directory must be outside the input directory to prevent recursive processing")
    if review_dir is not None:
        review = review_dir.resolve()
        if _is_within(review, source):
            raise ValueError("Review directory must be outside the input directory")
        if _is_within(review, output) or _is_within(output, review):
            raise ValueError("Review and output directories must be separate, non-overlapping directories")


def discover_images(input_dir: Path) -> List[Path]:
    """Return supported image paths in stable relative-path order."""
    paths = [
        path for path in input_dir.rglob("*")
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda path: path.relative_to(input_dir).as_posix().casefold())


def _score_list(detections: Sequence[Detection], detector_name: str) -> List[float]:
    return [round(float(item.scores[detector_name]), 6) for item in detections if detector_name in item.scores]


def _detection_dict(detection: Detection) -> Dict[str, object]:
    return {
        "box_xyxy": [round(float(value), 3) for value in detection.box],
        "detectors": list(detection.sources),
        "confidence_scores": {key: round(float(value), 6) for key, value in detection.scores.items()},
    }


def _read_yolo_boxes(label_path: Path, image_size: tuple, class_ids: Sequence[int]) -> List[tuple]:
    """Read selected YOLO xywh labels and convert them to clipped pixel xyxy boxes."""
    if not class_ids or not label_path.is_file():
        return []
    image_width, image_height = image_size
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if not values:
            continue
        if len(values) < 5:
            raise ValueError("Malformed YOLO label {}:{}".format(label_path, line_number))
        class_id = int(values[0])
        if class_id not in class_ids:
            continue
        center_x, center_y, width, height = map(float, values[1:5])
        left = max(0, math.floor((center_x - width / 2.0) * image_width))
        top = max(0, math.floor((center_y - height / 2.0) * image_height))
        right = min(image_width, math.ceil((center_x + width / 2.0) * image_width))
        bottom = min(image_height, math.ceil((center_y + height / 2.0) * image_height))
        if right > left and bottom > top:
            boxes.append((left, top, right, bottom))
    return boxes


def _filter_protected_overlaps(
    detections: Sequence[Detection],
    protected_boxes: Sequence[tuple],
    overlap_threshold: float,
) -> tuple:
    """Drop detections covering at least the requested fraction of a protected label."""
    kept = []
    excluded = 0
    for detection in detections:
        box = detection.box
        overlaps_protected_label = False
        for protected in protected_boxes:
            intersection_width = max(0.0, min(box[2], protected[2]) - max(box[0], protected[0]))
            intersection_height = max(0.0, min(box[3], protected[3]) - max(box[1], protected[1]))
            protected_area = max(0.0, protected[2] - protected[0]) * max(0.0, protected[3] - protected[1])
            intersection_area = intersection_width * intersection_height
            if protected_area and intersection_area and (intersection_area / protected_area) >= overlap_threshold:
                overlaps_protected_label = True
                break
        if overlaps_protected_label:
            excluded += 1
        else:
            kept.append(detection)
    return kept, excluded


def _restore_protected_pixels(image, original, protected_boxes: Sequence[tuple]):
    """Restore source pixels inside protected label boxes after face blurring."""
    restored = image.copy()
    for left, top, right, bottom in protected_boxes:
        restored.paste(original.crop((left, top, right, bottom)), (left, top))
    return restored


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".summary-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, str(path))
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _copy_support_files(input_dir: Path, output_dir: Path, overwrite: bool) -> List[str]:
    """Copy labels, YAML, and other non-image files without changing contents."""
    failures = []
    for source in sorted(input_dir.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not source.is_file() or source.is_symlink() or source.suffix.lower() in IMAGE_EXTENSIONS:
            continue
        relative = source.relative_to(input_dir)
        destination = output_dir / relative
        if destination.exists() and not overwrite:
            continue
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(destination))
        except OSError as exc:
            failures.append("{}: {}".format(relative.as_posix(), exc))
    return failures


def _record(
    relative_path: str,
    status: str,
    per_model: Mapping[str, Sequence[Detection]],
    final: Sequence[Detection],
    elapsed: float,
    error: str = "",
    excluded_protected: int = 0,
) -> Dict[str, object]:
    return {
        "relative_image_path": relative_path,
        "processing_status": status,
        "yunet_detections": len(per_model.get("yunet", [])),
        "scrfd_detections": len(per_model.get("scrfd", [])),
        "final_merged_detections": len(final),
        "yunet_confidence_scores": json.dumps(_score_list(per_model.get("yunet", []), "yunet")),
        "scrfd_confidence_scores": json.dumps(_score_list(per_model.get("scrfd", []), "scrfd")),
        "final_bounding_boxes_xyxy": json.dumps([
            [round(float(value), 3) for value in item.box] for item in final
        ]),
        "detectors_per_final_detection": json.dumps([_detection_dict(item) for item in final]),
        "detections_excluded_for_protected_label_overlap": excluded_protected,
        "processing_time_seconds": "{:.6f}".format(elapsed),
        "error_message": error,
    }


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    detector_mode: str = "both",
    yunet_model: Optional[Path] = None,
    scrfd_model: Optional[Path] = None,
    yunet_threshold: float = 0.5,
    scrfd_threshold: float = 0.4,
    padding: float = 0.30,
    blur_strength: float = 0.5,
    iou_threshold: float = 0.45,
    jpeg_quality: int = 100,
    protect_yolo_class_ids: Optional[Sequence[int]] = None,
    protected_overlap_threshold: float = 0.25,
    overwrite: bool = False,
    dry_run: bool = False,
    review_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    detector_instances: Optional[Mapping[str, object]] = None,
    show_progress: bool = True,
) -> Dict[str, object]:
    """Run face detection and anonymisation sequentially over one dataset tree."""
    if detector_mode not in ("yunet", "scrfd", "both"):
        raise ValueError("Detector must be yunet, scrfd, or both")
    if not 0.0 <= yunet_threshold <= 1.0 or not 0.0 <= scrfd_threshold <= 1.0:
        raise ValueError("Confidence thresholds must be between 0 and 1")
    if padding < 0 or padding > 2:
        raise ValueError("Padding must be between 0 and 2")
    if not 0.0 < blur_strength <= 1.0:
        raise ValueError("Blur strength must be greater than 0 and at most 1")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be between 0 and 1")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")
    if not 0.0 <= protected_overlap_threshold <= 1.0:
        raise ValueError("Protected-label overlap threshold must be between 0 and 1")
    protect_yolo_class_ids = sorted(set(protect_yolo_class_ids or []))
    if any(class_id < 0 for class_id in protect_yolo_class_ids):
        raise ValueError("Protected YOLO class IDs must be non-negative integers")
    if limit is not None and limit < 1:
        raise ValueError("Limit must be a positive integer")
    _validate_paths(input_dir, output_dir, review_dir)

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    review_dir = review_dir.resolve() if review_dir is not None else None
    start_time = time.perf_counter()
    all_images = discover_images(input_dir)
    selected = all_images[:limit] if limit is not None else all_images

    if detector_instances is None:
        script_dir = Path(__file__).resolve().parent
        yunet_model = yunet_model or script_dir / "models" / "face_detection_yunet_2023mar.onnx"
        scrfd_model = scrfd_model or script_dir / "models" / "scrfd_2.5g.onnx"
        detector_instances = create_detectors(
            detector_mode, yunet_model, scrfd_model, yunet_threshold, scrfd_threshold
        )
    required = {"yunet", "scrfd"} if detector_mode == "both" else {detector_mode}
    missing = required.difference(detector_instances)
    if missing:
        raise ValueError("Missing detector implementation(s): {}".format(", ".join(sorted(missing))))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    loop = tqdm(selected, desc="Anonymising BLPR images", unit="image", disable=not show_progress)
    for image_path in loop:
        image_start = time.perf_counter()
        relative = image_path.relative_to(input_dir)
        relative_text = relative.as_posix()
        destination = output_dir / relative
        per_model: Dict[str, List[Detection]] = {"yunet": [], "scrfd": []}
        final: List[Detection] = []
        excluded_protected = 0
        protected_boxes = []

        if not dry_run and destination.exists() and not overwrite:
            rows.append(_record(relative_text, "skipped_existing", per_model, final,
                                time.perf_counter() - image_start))
            continue

        try:
            asset = load_image(image_path)
        except Exception as exc:
            rows.append(_record(relative_text, "failed_read", per_model, final,
                                time.perf_counter() - image_start, str(exc)))
            continue

        try:
            rgb = np.asarray(asset.image.convert("RGB"))
            image_bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            for detector_name in ("yunet", "scrfd"):
                if detector_name in required:
                    per_model[detector_name] = detector_instances[detector_name].detect(image_bgr)
            combined = per_model["yunet"] + per_model["scrfd"]
            final = merge_detections(combined, iou_threshold) if detector_mode == "both" else list(combined)
            if protect_yolo_class_ids:
                label_path = input_dir / relative.parent.parent / "labels" / (relative.stem + ".txt")
                protected_boxes = _read_yolo_boxes(label_path, asset.image.size, protect_yolo_class_ids)
                final, excluded_protected = _filter_protected_overlaps(
                    final, protected_boxes, protected_overlap_threshold
                )
        except Exception as exc:
            rows.append(_record(relative_text, "failed_inference", per_model, final,
                                time.perf_counter() - image_start, str(exc)))
            continue

        if dry_run:
            status = "dry_run_faces_detected" if final else "dry_run_no_detection"
            rows.append(_record(relative_text, status, per_model, final,
                                time.perf_counter() - image_start,
                                excluded_protected=excluded_protected))
            continue

        try:
            if final:
                processed, _regions = blur_detections(asset.image, final, padding, blur_strength)
                if protected_boxes:
                    processed = _restore_protected_pixels(processed, asset.image, protected_boxes)
                save_image(
                    ImageAsset(processed, asset.image_format, asset.metadata),
                    destination,
                    overwrite,
                    jpeg_quality,
                )
                status = "anonymised"
                if review_dir is not None:
                    review_path = review_dir / relative.parent / (relative.stem + "_review.jpg")
                    try:
                        save_review_image(processed, final, review_path, padding, overwrite)
                    except Exception as exc:
                        review_error = "Review output failed: {}".format(exc)
                    else:
                        review_error = ""
                else:
                    review_error = ""
            else:
                copy_image_bytes(image_path, destination, overwrite)
                status = "no_detection"
                review_error = ""
            rows.append(_record(relative_text, status, per_model, final,
                                time.perf_counter() - image_start, review_error,
                                excluded_protected=excluded_protected))
        except Exception as exc:
            rows.append(_record(relative_text, "failed_save", per_model, final,
                                time.perf_counter() - image_start, str(exc)))

    support_copy_failures = [] if dry_run else _copy_support_files(input_dir, output_dir, overwrite)
    status_counts = {}
    for row in rows:
        status = str(row["processing_status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    successful = status_counts.get("anonymised", 0) + status_counts.get("no_detection", 0)
    dry_successful = status_counts.get("dry_run_faces_detected", 0) + status_counts.get("dry_run_no_detection", 0)
    with_faces = status_counts.get("anonymised", 0) + status_counts.get("dry_run_faces_detected", 0)
    without_faces = status_counts.get("no_detection", 0) + status_counts.get("dry_run_no_detection", 0)
    failures = sum(status_counts.get(status, 0) for status in ("failed_read", "failed_inference", "failed_save"))
    skipped = status_counts.get("skipped_existing", 0)
    anonymised_regions = sum(
        int(row["final_merged_detections"])
        for row in rows if row["processing_status"] == "anonymised"
    )
    excluded_protected_total = sum(int(row.get("detections_excluded_for_protected_label_overlap", 0)) for row in rows)
    exclusive_yunet = 0
    exclusive_scrfd = 0
    for row in rows:
        try:
            final_items = json.loads(str(row["detectors_per_final_detection"]))
        except (TypeError, ValueError):
            final_items = []
        for item in final_items:
            sources = set(item.get("detectors", []))
            if sources == {"yunet"}:
                exclusive_yunet += 1
            elif sources == {"scrfd"}:
                exclusive_scrfd += 1

    model_names = {}
    for name, detector_instance in detector_instances.items():
        model_path = getattr(detector_instance, "model_path", None)
        model_names[name] = Path(model_path).name if model_path else detector_instance.__class__.__name__
    summary = {
        "total_images_discovered": len(all_images),
        "images_selected_for_processing": len(selected),
        "successfully_processed_images": successful,
        "images_with_detected_faces": with_faces,
        "images_without_detected_faces": without_faces,
        "total_anonymised_face_regions": anonymised_regions,
        "images_with_processing_failures": failures,
        "images_skipped_because_outputs_exist": skipped,
        "faces_detected_exclusively_by_yunet": exclusive_yunet if detector_mode == "both" else 0,
        "faces_detected_exclusively_by_scrfd": exclusive_scrfd if detector_mode == "both" else 0,
        "dry_run_images_with_successful_inference": dry_successful,
        "support_file_copy_failures": support_copy_failures,
        "detections_excluded_for_protected_label_overlap": excluded_protected_total,
        "images_with_detections_excluded_for_protected_labels": sum(
            1 for row in rows if int(row.get("detections_excluded_for_protected_label_overlap", 0)) > 0
        ),
        "processing_status_counts": status_counts,
        "detector": detector_mode,
        "models": model_names,
        "parameters": {
            "yunet_threshold": yunet_threshold,
            "scrfd_threshold": scrfd_threshold,
            "padding_ratio_per_side": padding,
            "blur_strength": blur_strength,
            "iou_threshold": iou_threshold,
            "jpeg_quality": jpeg_quality,
            "protected_yolo_class_ids": protect_yolo_class_ids,
            "protected_label_overlap_threshold": protected_overlap_threshold,
            "overwrite": overwrite,
            "dry_run": dry_run,
            "limit": limit,
            "orientation_handling": "Use raw pixel array; do not apply EXIF transpose; retain EXIF metadata.",
        },
        "total_execution_time_seconds": round(time.perf_counter() - start_time, 6),
    }
    report_csv = output_dir / "face_anonymisation_report.csv"
    report_json = output_dir / "face_anonymisation_summary.json"
    if not overwrite and (report_csv.exists() or report_json.exists()):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_csv = output_dir / "face_anonymisation_report_{}.csv".format(stamp)
        report_json = output_dir / "face_anonymisation_summary_{}.json".format(stamp)
    summary["report_files"] = {"csv": report_csv.name, "json": report_json.name}
    _write_csv(report_csv, rows)
    _write_json(report_json, summary)
    return summary


def _unit_interval(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _jpeg_quality(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 100:
        raise argparse.ArgumentTypeError("must be between 1 and 100")
    return number


def _nonnegative_integer(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Source dataset directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Separate output dataset directory")
    parser.add_argument("--detector", choices=("yunet", "scrfd", "both"), default="both")
    parser.add_argument(
        "--yunet-model", type=Path,
        default=Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx",
        help="Path to the official OpenCV Zoo YuNet ONNX model",
    )
    parser.add_argument(
        "--scrfd-model", type=Path,
        default=Path(__file__).resolve().parent / "models" / "scrfd_2.5g.onnx",
        help="Path to the SCRFD detection ONNX file extracted from the official InsightFace buffalo_m pack",
    )
    parser.add_argument("--yunet-threshold", type=_unit_interval, default=0.5)
    parser.add_argument("--scrfd-threshold", type=_unit_interval, default=0.4)
    parser.add_argument("--padding", type=float, default=0.30, help="Expansion per side as a fraction of face width/height")
    parser.add_argument(
        "--blur-strength", type=float, default=0.5,
        help="Relative Gaussian kernel scale (0, 1]; 0.5 uses odd kernels about half the expanded-region dimensions",
    )
    parser.add_argument("--iou-threshold", type=_unit_interval, default=0.45)
    parser.add_argument(
        "--jpeg-quality", type=_jpeg_quality, default=100,
        help="JPEG encoding quality for images with detected faces (1-100; default: 100)",
    )
    parser.add_argument(
        "--protect-yolo-class-id", type=_nonnegative_integer, action="append", default=[],
        help="Protect source YOLO label boxes for this class ID; may be repeated",
    )
    parser.add_argument(
        "--protected-overlap-threshold", type=_unit_interval, default=0.25,
        help="Exclude a face detection when it covers at least this fraction of a protected label box",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing image and support-file outputs")
    parser.add_argument("--dry-run", action="store_true", help="Run detection and write reports, without saving images")
    parser.add_argument("--review-dir", type=Path, help="Separate directory for anonymised QC images")
    parser.add_argument("--limit", type=_positive_integer, help="Maximum number of images to process")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            detector_mode=args.detector,
            yunet_model=args.yunet_model,
            scrfd_model=args.scrfd_model,
            yunet_threshold=args.yunet_threshold,
            scrfd_threshold=args.scrfd_threshold,
            padding=args.padding,
            blur_strength=args.blur_strength,
            iou_threshold=args.iou_threshold,
            jpeg_quality=args.jpeg_quality,
            protect_yolo_class_ids=args.protect_yolo_class_id,
            protected_overlap_threshold=args.protected_overlap_threshold,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            review_dir=args.review_dir,
            limit=args.limit,
        )
    except (OSError, ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["images_with_processing_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
