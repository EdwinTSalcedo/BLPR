#!/usr/bin/env python3
"""Combine and anonymise the BLPR-D evaluation and calibration subsets."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import shutil
import sys
import time
import warnings
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

try:
    from .anonymise_faces import _filter_protected_overlaps, _record, _restore_protected_pixels
    from .detectors import Detection, create_detectors, intersection_over_union, merge_detections
    from .image_processing import ImageAsset, blur_detections, copy_image_bytes, expand_box, load_image, save_image
except ImportError:  # Direct script invocation.
    from anonymise_faces import _filter_protected_overlaps, _record, _restore_protected_pixels
    from detectors import Detection, create_detectors, intersection_over_union, merge_detections
    from image_processing import ImageAsset, blur_detections, copy_image_bytes, expand_box, load_image, save_image


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT.parent / "BLPR-datasets"
MODEL_ROOT = Path(__file__).resolve().parent / "models"
DEFAULT_ARCHIVE = DATA_ROOT / "BLPR-D" / "dataset_challenge_val.zip"
DEFAULT_CALIBRATION_ARCHIVE = DATA_ROOT / "BLPR-D" / "BLPR-D-Test.zip"
DEFAULT_ARCHIVES = (DEFAULT_ARCHIVE, DEFAULT_CALIBRATION_ARCHIVE)
DEFAULT_OUTPUT = DATA_ROOT / "BLPR-D-public-final"
HEIF_EXTENSIONS = {".heic", ".heif"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", *HEIF_EXTENSIONS}

# The explicit BLPR-A command in face_anonymisation/README.md, plus its defaults.
BLPR_A_PARAMETERS = {
    "yunet_threshold": 0.5,
    "scrfd_threshold": 0.4,
    "iou_threshold": 0.45,
    "padding": 0.30,
    "blur_strength": 0.5,
    "jpeg_quality": 95,
    "protected_overlap_threshold": 0.25,
}


@dataclass(frozen=True)
class ImageEntry:
    member: str
    original_name: str
    public_name: str
    source_archive: str
    subset: str


@dataclass
class ReleasePlan:
    images: list[ImageEntry]
    selected: list[ImageEntry]
    fields: list[str]
    rows: list[dict[str, str]]
    labels_source: str
    source_counts: list[dict[str, object]]


def normalize_label_field(field: str) -> str:
    normalized = field.strip().casefold()
    if normalized == "image":
        return "imagen"
    normalized = re.sub(r"^perspecvive(\d+)$", r"perspective\1", normalized)
    return normalized


def read_csv(text: str, description: str):
    reader = csv.DictReader(io.StringIO(text, newline=""))
    original_fields = reader.fieldnames
    if not original_fields:
        raise ValueError(f"{description}: expected a CSV header with an image column")
    fields = [normalize_label_field(field) for field in original_fields]
    if "imagen" not in fields or len(fields) != len(set(fields)):
        raise ValueError(f"{description}: expected unique CSV columns including 'imagen' or 'Image'")
    rows = list(reader)
    indexed = {}
    for line, row in enumerate(rows, 2):
        if not any(str(value or "").strip() for value in row.values()):
            continue  # Ignore blank trailing rows in exported spreadsheet CSVs.
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{description}:{line}: inconsistent CSV column count")
        row = {normalize_label_field(column): value for column, value in row.items()}
        key = (row.get("imagen") or "").strip()
        if not key or key in indexed:
            raise ValueError(f"{description}:{line}: empty or duplicate image reference {key!r}")
        row["imagen"] = key
        indexed[key] = row
    normalized_rows = list(indexed.values())
    return fields, normalized_rows, indexed


def build_plan(
    archive: ZipFile,
    labels_csv: Path | None = None,
    limit: int | None = None,
    subset: str = "",
) -> ReleasePlan:
    """Join on exact image names; extra companion columns are preserved verbatim."""
    members = [info.filename for info in archive.infolist() if not info.is_dir()]
    if len(members) != len(set(members)):
        raise ValueError("ZIP contains duplicate member paths")
    image_members = []
    csv_members = []
    basenames = set()
    for member in members:
        path = PurePosixPath(member)
        if path.is_absolute() or ".." in path.parts or "\\" in member:
            raise ValueError(f"Unsafe ZIP member path: {member}")
        if "__MACOSX" in path.parts or path.name.startswith("."):
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.name in basenames:
                raise ValueError(f"Ambiguous image basename in ZIP: {path.name}")
            basenames.add(path.name)
            image_members.append(member)
        elif path.suffix.lower() == ".csv":
            csv_members.append(member)
    if not image_members or len(csv_members) != 1:
        raise ValueError("ZIP must contain images and exactly one annotation CSV")

    csv_member = csv_members[0]
    fields, rows, archived = read_csv(archive.read(csv_member).decode("utf-8-sig"), csv_member)
    labels_source = f"{archive.filename}:{csv_member}"
    if labels_csv is not None:
        fields, rows, companion = read_csv(labels_csv.read_text(encoding="utf-8-sig"), str(labels_csv))
        if companion.keys() != archived.keys():
            raise ValueError("Companion CSV must reference exactly the same images as the archived CSV")
        for key, original in archived.items():
            for column, value in original.items():
                if companion[key].get(column) != value:
                    raise ValueError(f"Companion CSV changes archived value: {key}, {column}")
        labels_source = str(labels_csv.resolve())
    unknown = set(archived) - basenames
    if unknown:
        raise ValueError(f"CSV references images absent from ZIP: {sorted(unknown)}")

    source_archive = str(Path(archive.filename).resolve()) if archive.filename else ""
    images = []
    for number, member in enumerate(sorted(image_members), 1):
        original_name = PurePosixPath(member).name
        suffix = PurePosixPath(member).suffix.lower()
        output_suffix = ".jpg" if suffix in HEIF_EXTENSIONS else suffix
        images.append(ImageEntry(
            member, original_name, f"img{number}{output_suffix}", source_archive, subset,
        ))
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    selected = images[:limit] if limit is not None else images
    new_names = {entry.original_name: entry.public_name for entry in selected}
    order = {entry.public_name: number for number, entry in enumerate(selected)}
    renamed_rows = [dict(row, imagen=new_names[row["imagen"]]) for row in rows if row["imagen"] in new_names]
    renamed_rows.sort(key=lambda row: order[row["imagen"]])
    source_counts = [{
        "source_archive": source_archive,
        "subset": subset,
        "source_images": len(images),
        "label_rows": len(renamed_rows),
    }]
    return ReleasePlan(images, selected, fields, renamed_rows, labels_source, source_counts)


def subset_for_archive(source_archive: Path) -> str:
    name = source_archive.name.casefold()
    if name == DEFAULT_ARCHIVE.name.casefold():
        return "evaluation"
    if name == DEFAULT_CALIBRATION_ARCHIVE.name.casefold():
        return "calibration"
    return source_archive.stem


def build_combined_plan(
    archives: list[ZipFile],
    labels_csvs: list[Path | None],
    limit: int | None = None,
) -> ReleasePlan:
    """Join one or more source archives into a single numbered release plan."""
    if not archives or len(archives) != len(labels_csvs):
        raise ValueError("At least one source archive is required")
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")

    plans = []
    for archive, labels_csv in zip(archives, labels_csvs):
        source_path = Path(archive.filename) if archive.filename else Path("source.zip")
        plans.append(build_plan(archive, labels_csv, subset=subset_for_archive(source_path)))

    multiple_sources = len(plans) > 1
    fields = []
    if multiple_sources:
        fields.extend(("imagen", "subset"))
    for plan in plans:
        for field in plan.fields:
            if field not in fields:
                fields.append(field)
    if multiple_sources and any("subset" in source_plan.fields for source_plan in plans):
        raise ValueError("Source annotation CSV already uses the reserved 'subset' column")

    all_images = []
    local_names = []
    source_counts = []
    labels_sources = []
    for plan in plans:
        remap = {}
        source_archive = plan.source_counts[0]["source_archive"]
        subset = plan.source_counts[0]["subset"]
        for entry in plan.images:
            suffix = PurePosixPath(entry.public_name).suffix
            public_name = f"img{len(all_images) + 1}{suffix}"
            if entry.public_name in remap:
                raise ValueError(f"Duplicate local image identifier {entry.public_name!r} in {source_archive}")
            remap[entry.public_name] = public_name
            all_images.append(ImageEntry(
                entry.member, entry.original_name, public_name,
                entry.source_archive, entry.subset,
            ))
        local_names.append(remap)
        source_counts.extend(plan.source_counts)
        labels_sources.append(plan.labels_source)

    selected = all_images[:limit] if limit is not None else all_images
    selected_names = {entry.public_name for entry in selected}
    rows = []
    for plan, remap in zip(plans, local_names):
        subset = plan.source_counts[0]["subset"]
        for source_row in plan.rows:
            public_name = remap[source_row["imagen"]]
            if public_name not in selected_names:
                continue
            row = {field: "" for field in fields}
            row.update({field: value for field, value in source_row.items() if field in fields})
            row["imagen"] = public_name
            if multiple_sources:
                row["subset"] = subset
            rows.append(row)
    order = {entry.public_name: index for index, entry in enumerate(selected)}
    rows.sort(key=lambda row: order[row["imagen"]])
    labels_source = labels_sources[0] if len(labels_sources) == 1 else "; ".join(labels_sources)
    return ReleasePlan(all_images, selected, fields, rows, labels_source, source_counts)


def companion_matches_archive(archive: ZipFile, candidate: Path) -> bool:
    """Only auto-use a neighboring CSV when it refers to this archive's images."""
    csv_members = [
        info.filename for info in archive.infolist()
        if not info.is_dir() and PurePosixPath(info.filename).suffix.lower() == ".csv"
        and "__MACOSX" not in PurePosixPath(info.filename).parts
    ]
    if len(csv_members) != 1:
        return False
    try:
        _, _, embedded = read_csv(archive.read(csv_members[0]).decode("utf-8-sig"), csv_members[0])
        _, _, companion = read_csv(candidate.read_text(encoding="utf-8-sig"), str(candidate))
    except (OSError, UnicodeError, ValueError):
        return False
    return embedded.keys() == companion.keys()


def read_dict_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def load_reusable_release(
    public_dir: Path,
    source_archive: Path,
    expected_images: list[ImageEntry],
    expected_labels: dict[str, dict[str, str]],
    yunet_model: Path,
    scrfd_model: Path,
    plate_model: Path,
    plate_confidence: float,
    tile_size: int,
    tile_overlap: float,
    detection_strategy: str,
    require_review: bool,
):
    """Load a completed, compatible public subset for byte-for-byte reuse."""
    audit_dir = public_dir.with_name(public_dir.name + "-audit")
    summary_path = audit_dir / "summary.json"
    mapping_path = audit_dir / "filename_map.csv"
    report_path = audit_dir / "face_anonymisation_report.csv"
    if not public_dir.is_dir() or not summary_path.is_file() or not mapping_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"Reusable release and complete audit files are required: {public_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError(f"Reusable release is not complete: {summary.get('status')!r}")
    recorded_source = summary.get("source_zip")
    if not recorded_source or Path(recorded_source).resolve() != source_archive.resolve():
        raise ValueError("Reusable release was prepared from a different source ZIP")
    expected_names = {entry.original_name for entry in expected_images}
    if summary.get("processed_images") != len(expected_images) or summary.get("source_images") != len(expected_images):
        raise ValueError("Reusable release image count does not match the source archive")
    if summary.get("face_parameters") != BLPR_A_PARAMETERS:
        raise ValueError("Reusable release does not use the current BLPR-A face parameters")
    expected_detection = {"strategy": detection_strategy, "max_side": tile_size, "tile_overlap": tile_overlap}
    if summary.get("detection") != expected_detection:
        raise ValueError("Reusable release uses different face-detection settings")
    old_models = summary.get("models", {})
    expected_models = {"yunet": str(yunet_model.resolve()), "scrfd": str(scrfd_model.resolve())}
    if old_models != expected_models:
        raise ValueError("Reusable release was processed with different face detector models")
    old_plate = summary.get("plate_protection", {})
    if (
        old_plate.get("class_id") != 1
        or old_plate.get("confidence") != plate_confidence
        or old_plate.get("imgsz") != 640
        or Path(old_plate.get("model", "")).resolve() != plate_model.resolve()
    ):
        raise ValueError("Reusable release was processed with different plate-protection settings")

    map_rows = read_dict_csv(mapping_path)
    if not {"original_name", "public_name"}.issubset(map_rows[0] if map_rows else {}):
        raise ValueError("Reusable filename map is missing required columns")
    old_names = {}
    for row in map_rows:
        if row["original_name"] in old_names:
            raise ValueError(f"Duplicate original name in reusable filename map: {row['original_name']}")
        old_names[row["original_name"]] = row["public_name"]
    if set(old_names) != expected_names:
        raise ValueError("Reusable filename map does not cover exactly the expected source images")
    _, old_label_rows, _ = read_csv(
        (public_dir / "labels.csv").read_text(encoding="utf-8-sig"), str(public_dir / "labels.csv"),
    )
    old_to_original = {public_name: original_name for original_name, public_name in old_names.items()}
    old_labels_by_original = {
        old_to_original[row["imagen"]]: row for row in old_label_rows
        if row["imagen"] in old_to_original
    }
    if set(old_labels_by_original) != set(expected_labels):
        raise ValueError("Reusable labels do not cover the same annotated source images")
    for original_name, expected_row in expected_labels.items():
        previous = old_labels_by_original[original_name]
        for field, value in expected_row.items():
            if field not in {"imagen", "subset"} and previous.get(field, "") != value:
                raise ValueError(f"Reusable labels differ from the current source CSV: {original_name}, {field}")
    for original_name, public_name in old_names.items():
        if not (public_dir / public_name).is_file():
            raise FileNotFoundError(f"Reusable output image is missing: {public_dir / public_name}")

    report_rows = read_dict_csv(report_path)
    records = {}
    for row in report_rows:
        key = row.get("relative_image_path", "")
        if not key or key in records:
            raise ValueError(f"Missing or duplicate image path in reusable report: {key!r}")
        if row.get("error_message"):
            raise ValueError(f"Reusable report contains a failed image: {key}")
        for field in (
            "yunet_detections", "scrfd_detections", "final_merged_detections",
            "detections_excluded_for_protected_label_overlap",
        ):
            try:
                row[field] = int(row[field])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Reusable report has an invalid {field} value for {key}") from error
        records[key] = row
    if set(records) != set(old_names.values()):
        raise ValueError("Reusable anonymisation report does not cover exactly the mapped images")
    if require_review:
        review_dir = audit_dir / "review"
        missing = [name for name in old_names.values() if not (review_dir / f"{Path(name).stem}.jpg").is_file()]
        if missing:
            raise FileNotFoundError("Reusable release lacks review images; omit --review or prepare the source with --review")

    return {"public_dir": public_dir, "audit_dir": audit_dir, "old_names": old_names, "records": records}


class PlateDetector:
    """Use the existing BLPR plate model to supply protection boxes, not new labels."""

    def __init__(self, model_path: Path, confidence: float = 0.5):
        if not model_path.is_file():
            raise FileNotFoundError(f"Plate model not found: {model_path}")
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        if str(self.model.names[1]).casefold() not in {"plate", "placa", "license plate", "licence plate"}:
            raise ValueError("Plate model class 1 must represent a plate, not a vehicle")
        self.confidence = confidence

    def detect(self, image_bgr):
        detections = []
        for result in self.model.predict(
            image_bgr, classes=[1], conf=self.confidence, iou=0.45,
            imgsz=640, device="cpu", verbose=False, save=False,
        ):
            for box in result.boxes:
                detections.append(Detection(
                    tuple(float(x) for x in box.xyxy[0].cpu().tolist()),
                    {"plate": float(box.conf[0].cpu())},
                ))
        return detections


def tile_origins(length: int, tile_size: int, overlap: float) -> list[int]:
    last = max(0, length - tile_size)
    origins = list(range(0, last + 1, max(1, int(tile_size * (1 - overlap)))))
    if origins[-1] != last:
        origins.append(last)
    return origins


def detection_views(image: Image.Image, tile_size: int, overlap: float, include_tiles: bool = False):
    """Rescale once for fast detection, or add native-scale tiles for small faces."""
    width, height = image.size
    if max(image.size) > tile_size:
        ratio = tile_size / max(image.size)
        overview = image.resize((max(1, round(width * ratio)), max(1, round(height * ratio))), Image.Resampling.LANCZOS)
        try:
            yield overview, 0, 0, width / overview.width, height / overview.height
        finally:
            overview.close()
    elif not include_tiles:
        yield image, 0, 0, 1.0, 1.0
    if not include_tiles:
        return
    for top in tile_origins(height, tile_size, overlap):
        for left in tile_origins(width, tile_size, overlap):
            tile = image.crop((left, top, min(width, left + tile_size), min(height, top + tile_size)))
            try:
                yield tile, left, top, 1.0, 1.0
            finally:
                tile.close()


def deduplicate(detections: list[Detection]) -> list[Detection]:
    """Suppress duplicate tile detections within a model, retaining isolated faces."""
    kept = []
    for item in sorted(detections, key=lambda detection: -max(detection.scores.values())):
        if not any(intersection_over_union(item.box, previous.box) >= BLPR_A_PARAMETERS["iou_threshold"] for previous in kept):
            kept.append(item)
    return kept


def detect_image(image, face_detectors, plate_detector, tile_size: int, tile_overlap: float, include_tiles: bool = False):
    per_model = {"yunet": [], "scrfd": [], "plate": []}
    models = dict(face_detectors, plate=plate_detector)
    width, height = image.size
    for view, left, top, scale_x, scale_y in detection_views(image, tile_size, tile_overlap, include_tiles):
        bgr = np.ascontiguousarray(np.asarray(view)[:, :, ::-1])
        for name, model in models.items():
            for detection in model.detect(bgr):
                x1, y1, x2, y2 = detection.box
                values = (left + x1 * scale_x, top + y1 * scale_y, left + x2 * scale_x, top + y2 * scale_y)
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"{name} produced non-finite coordinates")
                box = (max(0.0, values[0]), max(0.0, values[1]), min(float(width), values[2]), min(float(height), values[3]))
                if box[2] > box[0] and box[3] > box[1]:
                    per_model[name].append(Detection(box, detection.scores))
        del bgr
    per_model = {name: deduplicate(items) for name, items in per_model.items()}
    protected = [
        (math.floor(item.box[0]), math.floor(item.box[1]), math.ceil(item.box[2]), math.ceil(item.box[3]))
        for item in per_model["plate"]
    ]
    merged = merge_detections(per_model["yunet"] + per_model["scrfd"], BLPR_A_PARAMETERS["iou_threshold"])
    kept, excluded_count = _filter_protected_overlaps(merged, protected, BLPR_A_PARAMETERS["protected_overlap_threshold"])
    excluded = [item for item in merged if item not in kept]
    assert excluded_count == len(excluded)
    return per_model, kept, excluded, protected


def write_review(image, kept, excluded, protected, destination: Path):
    review = image.copy()
    review.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(review)
    sx, sy = review.width / image.width, review.height / image.height

    def rectangle(box, color):
        draw.rectangle((box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy), outline=color, width=2)

    for box in protected:
        rectangle(box, "yellow")
    for face in kept:
        rectangle(expand_box(face.box, BLPR_A_PARAMETERS["padding"], image.size), "lime")
    for face in excluded:
        rectangle(face.box, "red")
    draw.text((10, 10), "Green: blurred face | Yellow: protected plate | Red: excluded face detection", fill="lime")
    destination.parent.mkdir(parents=True, exist_ok=True)
    review.save(destination, format="JPEG", quality=90)
    review.close()


def write_csv(path: Path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare_release(
    source_zip: Path | list[Path] | tuple[Path, ...],
    output_dir: Path,
    labels_csv: Path | None = None,
    archive_labels_only: bool = False,
    yunet_model: Path = MODEL_ROOT / "face_detection_yunet_2023mar.onnx",
    scrfd_model: Path = MODEL_ROOT / "det_2.5g.onnx",
    plate_model: Path = REPO_ROOT / "models" / "best_lpd.pt",
    plate_confidence: float = 0.5,
    tile_size: int = 2048,
    tile_overlap: float = 0.25,
    detection_strategy: str = "overview",
    limit: int | None = None,
    dry_run: bool = False,
    review: bool = False,
    reuse_public_dir: Path | None = None,
    face_detectors=None,
    plate_detector=None,
    show_progress: bool = True,
):
    if tile_size < 64 or not 0 <= tile_overlap < 1 or not 0 <= plate_confidence <= 1:
        raise ValueError("Require tile-size >= 64, 0 <= tile-overlap < 1, and 0 <= plate-confidence <= 1")
    if detection_strategy not in {"overview", "tiled"}:
        raise ValueError("Detection strategy must be 'overview' or 'tiled'")
    if archive_labels_only and labels_csv is not None:
        raise ValueError("Choose either --labels-csv or --archive-labels-only")
    source_paths = [source_zip] if isinstance(source_zip, (str, Path)) else list(source_zip)
    if not source_paths:
        raise ValueError("At least one source ZIP is required")
    source_paths = [Path(path).resolve() for path in source_paths]
    if len(source_paths) != len(set(source_paths)):
        raise ValueError("The same source ZIP was specified more than once")
    if len({path.name for path in source_paths}) != len(source_paths):
        raise ValueError("Source ZIP basenames must be unique")
    output_dir = output_dir.resolve()
    audit_dir = output_dir.with_name(output_dir.name + "-audit")
    partial_dir = output_dir.with_name(output_dir.name + "-partial")

    with ExitStack() as stack:
        archives = [stack.enter_context(ZipFile(path)) for path in source_paths]
        labels_csvs = []
        for index, archive in enumerate(archives):
            candidate = None
            if index == 0 and not archive_labels_only:
                if labels_csv is not None:
                    candidate = labels_csv.resolve()
                else:
                    sibling_csv = source_paths[index].parent / "labels.csv"
                    if sibling_csv.is_file() and companion_matches_archive(archive, sibling_csv):
                        candidate = sibling_csv
            labels_csvs.append(candidate)
        plan = build_combined_plan(archives, labels_csvs, limit)
        reusable = None
        if reuse_public_dir is not None:
            primary_images = [entry for entry in plan.images if entry.source_archive == str(source_paths[0])]
            primary_public_names = {entry.public_name: entry.original_name for entry in primary_images}
            expected_primary_labels = {
                primary_public_names[row["imagen"]]: row
                for row in plan.rows
                if row["imagen"] in primary_public_names
            }
            reusable = load_reusable_release(
                reuse_public_dir.resolve(), source_paths[0], primary_images, expected_primary_labels,
                yunet_model, scrfd_model, plate_model, plate_confidence,
                tile_size, tile_overlap, detection_strategy, review,
            )
        reusable_names = reusable["old_names"] if reusable else {}
        reusable_source = str(source_paths[0]) if reusable else ""
        selected_reused = sum(
            entry.source_archive == reusable_source and entry.original_name in reusable_names
            for entry in plan.selected
        ) if reusable else 0
        summary = {
            "source_zip": str(source_paths[0]),
            "source_zips": [str(path) for path in source_paths],
            "source_counts": plan.source_counts,
            "labels_source": plan.labels_source,
            "source_images": len(plan.images), "selected_images": len(plan.selected),
            "label_rows": len(plan.rows), "output_dir": str(output_dir),
            "audit_dir": str(audit_dir), "preview": len(plan.selected) < len(plan.images),
            "reused_images": selected_reused,
            "face_parameters": BLPR_A_PARAMETERS,
            "plate_protection": {"source": "predicted boxes", "model": str(plate_model), "class_id": 1, "confidence": plate_confidence, "imgsz": 640},
            "models": {"yunet": str(yunet_model), "scrfd": str(scrfd_model)},
            "detection": {"strategy": detection_strategy, "max_side": tile_size, "tile_overlap": tile_overlap},
            "orientation": "raw pixels; no EXIF transpose; metadata preserved, as in BLPR-A",
        }
        if dry_run:
            return dict(summary, status="dry_run")
        for path in (output_dir, audit_dir, partial_dir):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing path: {path}")
        pending = [
            entry for entry in plan.selected
            if not (
                reusable
                and entry.source_archive == reusable_source
                and entry.original_name in reusable_names
            )
        ]
        if any(PurePosixPath(entry.original_name).suffix.lower() in HEIF_EXTENSIONS for entry in pending):
            try:
                from pillow_heif import register_heif_opener
            except ImportError as error:
                raise RuntimeError("HEIC inputs require pillow-heif; install it with `pip install pillow-heif`") from error
            register_heif_opener()
        if pending and face_detectors is None:
            face_detectors = create_detectors(
                "both", yunet_model, scrfd_model,
                BLPR_A_PARAMETERS["yunet_threshold"], BLPR_A_PARAMETERS["scrfd_threshold"],
            )
        if pending and set(face_detectors) != {"yunet", "scrfd"}:
            raise ValueError("Both YuNet and SCRFD are required")
        if pending and plate_detector is None:
            plate_detector = PlateDetector(plate_model, plate_confidence)

        data_stage, audit_stage = partial_dir / "data", partial_dir / "audit"
        data_stage.mkdir(parents=True)
        audit_stage.mkdir()
        source_image = partial_dir / "source-image"
        reports = []
        started = time.perf_counter()
        summary["status"] = "processing"
        write_json(audit_stage / "summary.json", summary)
        write_csv(audit_stage / "filename_map.csv", ["source_archive", "subset", "original_name", "public_name"], [
            {
                "source_archive": Path(entry.source_archive).name,
                "subset": entry.subset,
                "original_name": entry.original_name,
                "public_name": entry.public_name,
            }
            for entry in plan.selected
        ])
        archive_by_path = {
            str(Path(archive.filename).resolve()): archive
            for archive in archives if archive.filename
        }
        current = None
        try:
            for entry in tqdm(plan.selected, desc="Preparing BLPR-D", unit="image", disable=not show_progress):
                current = entry
                if (
                    reusable
                    and entry.source_archive == reusable_source
                    and entry.original_name in reusable_names
                ):
                    old_public_name = reusable_names[entry.original_name]
                    shutil.copy2(reusable["public_dir"] / old_public_name, data_stage / entry.public_name)
                    record = dict(reusable["records"][old_public_name])
                    record.update({
                        "relative_image_path": entry.public_name,
                        "original_image_name": entry.original_name,
                        "source_archive": Path(entry.source_archive).name,
                        "subset": entry.subset,
                    })
                    reports.append(record)
                    if review:
                        old_review = reusable["audit_dir"] / "review" / f"{Path(old_public_name).stem}.jpg"
                        new_review = audit_stage / "review" / f"{Path(entry.public_name).stem}.jpg"
                        new_review.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(old_review, new_review)
                    with (audit_stage / "progress.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                    continue

                image_start = time.perf_counter()
                archive = archive_by_path[entry.source_archive]
                with archive.open(entry.member) as source, source_image.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                    with Image.open(source_image) as header:
                        if header.width * header.height > 150_000_000:
                            raise ValueError(f"Image exceeds the 150-megapixel processing limit: {entry.original_name}")
                    asset = load_image(source_image)
                processed = None
                try:
                    per_model, kept, excluded, protected = detect_image(
                        asset.image, face_detectors, plate_detector, tile_size, tile_overlap,
                        include_tiles=detection_strategy == "tiled",
                    )
                    destination = data_stage / entry.public_name
                    transcode_to_jpeg = PurePosixPath(entry.original_name).suffix.lower() in HEIF_EXTENSIONS
                    output_format = "JPEG" if transcode_to_jpeg else asset.image_format
                    if kept:
                        processed, regions = blur_detections(asset.image, kept, BLPR_A_PARAMETERS["padding"], BLPR_A_PARAMETERS["blur_strength"])
                        if protected:
                            restored = _restore_protected_pixels(processed, asset.image, protected)
                            processed.close()
                            processed = restored
                        save_image(ImageAsset(processed, output_format, asset.metadata), destination, jpeg_quality=BLPR_A_PARAMETERS["jpeg_quality"])
                        status = "anonymised"
                    else:
                        regions = []
                        if transcode_to_jpeg:
                            save_image(ImageAsset(asset.image, "JPEG", asset.metadata), destination, jpeg_quality=BLPR_A_PARAMETERS["jpeg_quality"])
                        else:
                            copy_image_bytes(source_image, destination)
                        status = "no_detection"
                    if review:
                        write_review(processed if processed is not None else asset.image, kept, excluded, protected, audit_stage / "review" / f"{Path(entry.public_name).stem}.jpg")
                    record = _record(entry.public_name, status, per_model, kept, time.perf_counter() - image_start, excluded_protected=len(excluded))
                    record.update({
                        "original_image_name": entry.original_name,
                        "source_archive": Path(entry.source_archive).name,
                        "subset": entry.subset,
                        "source_format": PurePosixPath(entry.original_name).suffix.lower().lstrip("."),
                        "output_format": output_format.casefold(),
                        "protected_plate_boxes_xyxy": json.dumps(protected),
                        "predicted_plate_detections": json.dumps([
                            {"box_xyxy": detection.box, "confidence": detection.scores["plate"]}
                            for detection in per_model["plate"]
                        ]),
                        "excluded_face_boxes_xyxy": json.dumps([d.box for d in excluded]),
                        "blur_regions_xyxy": json.dumps(regions),
                        "width": asset.image.width, "height": asset.image.height,
                    })
                    reports.append(record)
                    with (audit_stage / "progress.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                finally:
                    asset.image.close()
                    if processed is not None:
                        processed.close()
                    source_image.unlink(missing_ok=True)
            write_csv(data_stage / "labels.csv", plan.fields, plan.rows)
            report_fields = []
            for report in reports:
                for field in report:
                    if field not in report_fields:
                        report_fields.append(field)
            write_csv(audit_stage / "face_anonymisation_report.csv", report_fields, reports)
            summary.update({
                "status": "preview_complete" if summary["preview"] else "complete",
                "processed_images": len(reports),
                "newly_processed_images": len(reports) - selected_reused,
                "reused_images": selected_reused,
                "images_anonymised": sum(row["processing_status"] == "anonymised" for row in reports),
                "images_without_retained_detections": sum(row["processing_status"] == "no_detection" for row in reports),
                "face_regions_blurred": sum(row["final_merged_detections"] for row in reports),
                "detections_excluded_for_plate_overlap": sum(row["detections_excluded_for_protected_label_overlap"] for row in reports),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            })
            write_json(audit_stage / "summary.json", summary)
            audit_stage.rename(audit_dir)
            data_stage.rename(output_dir)
            partial_dir.rmdir()
            return summary
        except BaseException as error:
            source_image.unlink(missing_ok=True)
            summary.update(status="failed", error=str(error) or type(error).__name__, failed_image=current.original_name if current else None, processed_images=len(reports))
            failure_audit = audit_stage if audit_stage.exists() else audit_dir
            write_json(failure_audit / "summary.json", summary)
            print(f"Incomplete run retained at {partial_dir}; final release was not completed.", file=sys.stderr)
            raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-zip", type=Path, action="append",
        help="Source ZIP; repeat for multiple archives. Defaults to the BLPR-D evaluation and calibration archives.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    csv_options = parser.add_mutually_exclusive_group()
    csv_options.add_argument("--labels-csv", type=Path, help="Full companion CSV for the first source; by default a matching sibling labels.csv is used")
    csv_options.add_argument("--archive-labels-only", action="store_true", help="Use only the CSV embedded in the ZIP")
    parser.add_argument("--yunet-model", type=Path, default=MODEL_ROOT / "face_detection_yunet_2023mar.onnx")
    parser.add_argument("--scrfd-model", type=Path, default=MODEL_ROOT / "det_2.5g.onnx")
    parser.add_argument("--plate-model", type=Path, default=REPO_ROOT / "models" / "best_lpd.pt")
    parser.add_argument("--plate-confidence", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=2048, help="Detection overview side, or tile side with --detection-strategy tiled")
    parser.add_argument("--tile-overlap", type=float, default=0.25, help="Overlap between full-resolution tiles")
    parser.add_argument("--detection-strategy", choices=("overview", "tiled"), default="overview", help="Fast resized overview (default) or slower native-resolution tiles")
    parser.add_argument("--limit", type=int, help="Build a preview using the first N images; numbering remains consistent with the full run")
    parser.add_argument("--dry-run", action="store_true", help="Inspect CSV joins and report the plan without loading models or writing files")
    parser.add_argument("--review", action="store_true", help="Save annotated QC previews in the separate audit folder")
    parser.add_argument(
        "--reuse-public-dir", type=Path,
        help="Reuse a completed anonymised release for the first source ZIP, after checking its audit settings",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.source_zip is None:
        args.source_zip = list(DEFAULT_ARCHIVES)
    try:
        summary = prepare_release(**vars(args))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
