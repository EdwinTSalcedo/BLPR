#!/usr/bin/env python3
"""Create a contact sheet of random BLPR-D source and anonymised image pairs."""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from datetime import datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
import random
import sys
import warnings
from zipfile import BadZipFile, ZipFile

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT.parent / "BLPR-datasets"
DEFAULT_SOURCE_ZIP = DATA_ROOT / "BLPR-D" / "dataset_challenge_val.zip"
DEFAULT_CALIBRATION_ZIP = DATA_ROOT / "BLPR-D" / "BLPR-D-Test.zip"
DEFAULT_SOURCE_ZIPS = (DEFAULT_SOURCE_ZIP, DEFAULT_CALIBRATION_ZIP)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "BLPR-D-public-final"
HEIF_EXTENSIONS = {".heic", ".heif"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", *HEIF_EXTENSIONS}
THUMB_SIZE = (360, 270)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-zip", type=Path, action="append",
        help="Source ZIP; repeat for multiple archives. Defaults to both BLPR-D subsets.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit-dir", type=Path, help="Defaults to a sibling named <output>-audit")
    parser.add_argument("--samples", type=int, default=5, help="Number of random pairs (default: 5)")
    parser.add_argument("--seed", type=int, help="Optional seed to repeat a sample")
    parser.add_argument("--output", type=Path, help="PNG path; defaults to a timestamped file in the audit folder")
    return parser.parse_args(argv)


def read_csv(path: Path, key: str):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {}
    for row in rows:
        value = row.get(key, "")
        if not value or value in indexed:
            raise ValueError(f"Missing or duplicate {key!r} value in {path}: {value!r}")
        indexed[value] = row
    return indexed


def thumbnail(image: Image.Image) -> Image.Image:
    fitted = ImageOps.contain(image.convert("RGB"), THUMB_SIZE, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", THUMB_SIZE, (245, 245, 245))
    canvas.paste(fitted, ((THUMB_SIZE[0] - fitted.width) // 2, (THUMB_SIZE[1] - fitted.height) // 2))
    return canvas


def open_zip_thumbnail(archive: ZipFile, member: str) -> Image.Image:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(BytesIO(archive.read(member))) as image:
            return thumbnail(image)


def open_file_thumbnail(path: Path) -> Image.Image:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            return thumbnail(image)


def make_sheet(source_zips: list[Path], output_dir: Path, audit_dir: Path, sample_count: int, seed: int | None, output: Path | None):
    if sample_count < 1:
        raise ValueError("--samples must be at least 1")
    for required in (output_dir / "labels.csv", audit_dir / "filename_map.csv", audit_dir / "face_anonymisation_report.csv"):
        if not required.is_file():
            raise FileNotFoundError(f"Required release file not found: {required}")

    mapping = read_csv(audit_dir / "filename_map.csv", "public_name")
    records = read_csv(audit_dir / "face_anonymisation_report.csv", "relative_image_path")
    labels = read_csv(output_dir / "labels.csv", "imagen")
    unknown_label_images = set(labels) - set(mapping)
    if unknown_label_images:
        raise ValueError(f"labels.csv references images absent from the filename map: {sorted(unknown_label_images)[:5]}")
    available = sorted(
        public_name for public_name in mapping
        if (output_dir / public_name).is_file() and public_name in records
    )
    if not available:
        raise ValueError("No matching processed images are available")
    selected = random.Random(seed).sample(available, min(sample_count, len(available)))

    with ExitStack() as stack:
        archives = {}
        members_by_archive = {}
        for source_zip in source_zips:
            source_zip = source_zip.resolve()
            if source_zip.name in archives:
                raise ValueError(f"Source ZIP basenames must be unique: {source_zip.name}")
            archive = stack.enter_context(ZipFile(source_zip))
            archives[source_zip.name] = archive
            names_to_members = {}
            for member in archive.namelist():
                name = PurePosixPath(member).name
                if PurePosixPath(member).suffix.lower() in IMAGE_EXTENSIONS:
                    if name in names_to_members:
                        raise ValueError(f"Duplicate source image basename in {source_zip.name}: {name}")
                    names_to_members[name] = member
            members_by_archive[source_zip.name] = names_to_members

        if any(
            PurePosixPath(member).suffix.lower() in HEIF_EXTENSIONS
            for files in members_by_archive.values() for member in files.values()
        ):
            try:
                from pillow_heif import register_heif_opener
            except ImportError as error:
                raise RuntimeError("HEIC source images require pillow-heif; install it with `pip install pillow-heif`") from error
            register_heif_opener()

        cols, thumb_w, thumb_h = 4, *THUMB_SIZE
        margin, gap, header_h, caption_h = 24, 18, 82, 48
        row_h = caption_h + 24 + thumb_h
        rows = len(selected)
        sheet_w = margin * 2 + cols * thumb_w + (cols - 1) * gap
        sheet_h = header_h + margin + rows * row_h + (rows - 1) * gap + margin
        sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.load_default(size=16)
        except TypeError:  # Pillow before sized default fonts.
            font = ImageFont.load_default()
        draw.text((margin, 20), "BLPR-D random anonymisation review", fill=(20, 20, 20), font=font)
        draw.text(
            (margin, 46),
            f"{len(available)} processed images | showing {len(selected)} | "
            f"seed: {seed if seed is not None else 'random each run'}",
            fill=(70, 70, 70), font=font,
        )

        for row_no, public_name in enumerate(selected):
            image_map = mapping[public_name]
            original_name = image_map["original_name"]
            source_archive = Path(image_map.get("source_archive") or source_zips[0].name).name
            if source_archive not in archives:
                raise ValueError(f"Supply the source ZIP recorded for {public_name}: {source_archive}")
            names_to_members = members_by_archive.get(source_archive, {})
            member = names_to_members.get(original_name)
            if member is None:
                raise ValueError(f"Original image is absent from {source_archive}: {original_name}")
            record = records[public_name]
            source_thumb = open_zip_thumbnail(archives[source_archive], member)
            public_thumb = open_file_thumbnail(output_dir / public_name)
            diff = ImageChops.difference(source_thumb, public_thumb)
            diff_view = ImageOps.autocontrast(diff) if diff.getbbox() else diff

            review_path = audit_dir / "review" / f"{Path(public_name).stem}.jpg"
            review_saved = review_path.is_file()
            review_thumb = open_file_thumbnail(review_path) if review_saved else Image.new("RGB", THUMB_SIZE, (238, 238, 238))
            caption = (
                f"{public_name} | {record.get('processing_status', 'unknown')} | "
                f"blurred: {record.get('final_merged_detections', '?')} | "
                f"plate-overlap exclusions: {record.get('detections_excluded_for_protected_label_overlap', '?')}"
            )
            if record.get("subset"):
                caption += f" | {record['subset']}"
            x0 = margin
            y0 = header_h + margin + row_no * (row_h + gap)
            draw.text((x0, y0), caption, fill=(25, 25, 25), font=font)
            panels = (
                ("Source", source_thumb),
                ("Public output", public_thumb),
                ("Changes vs source (JPEG encoding included)", diff_view),
                ("Detector review" if review_saved else "Review not saved", review_thumb),
            )
            for col, (title, image) in enumerate(panels):
                x = x0 + col * (thumb_w + gap)
                y = y0 + 24
                draw.text((x, y), title, fill=(55, 55, 55), font=font)
                top = y + 20
                sheet.paste(image, (x, top))
                draw.rectangle((x, top, x + thumb_w - 1, top + thumb_h - 1), outline=(150, 150, 150))
            source_thumb.close()
            public_thumb.close()
            diff.close()
            if diff_view is not diff:
                diff_view.close()
            review_thumb.close()

        if output is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output = audit_dir / f"random_samples_{stamp}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output, format="PNG")
        sheet.close()
    print(f"Saved {len(selected)} random sample pairs to {output}")
    if len(selected) < sample_count:
        print(f"Only {len(selected)} completed image pairs were available.")
    return output


def main(argv=None):
    args = parse_args(argv)
    args.source_zip = args.source_zip or list(DEFAULT_SOURCE_ZIPS)
    audit_dir = args.audit_dir or args.output_dir.with_name(args.output_dir.name + "-audit")
    try:
        make_sheet(args.source_zip, args.output_dir, audit_dir, args.samples, args.seed, args.output)
    except (OSError, BadZipFile, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
