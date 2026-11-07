import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from face_anonymisation.detectors import Detection
from face_anonymisation.prepare_blpr_d_public import build_plan, detect_image, prepare_release


class Detector:
    def __init__(self, detections=(), error=None):
        self.detections = detections
        self.error = error
        self.sizes = []

    def detect(self, image):
        self.sizes.append(image.shape[:2])
        if self.error:
            raise RuntimeError(self.error)
        return list(self.detections)


def csv_text(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


class PublicReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archive = self.root / "dataset_challenge_val.zip"
        self.output = self.root / "public"
        yy, xx = np.indices((64, 64))
        pixels = ((xx + yy) % 2 * 255).astype(np.uint8)
        self.pixels = np.stack([pixels] * 3, axis=-1)
        self.buffer = io.BytesIO()
        Image.fromarray(self.pixels).save(self.buffer, format="PNG")

    def tearDown(self):
        self.temp.cleanup()

    def make_archive(self, names=("a.png", "b.png", "c.png"), labeled=("b.png", "c.png")):
        rows = [{"imagen": name, "light": "10000", "plate1": "1234ABC"} for name in labeled]
        with ZipFile(self.archive, "w") as archive:
            # Different ZIP order must not affect the numeric image identifiers.
            for name in reversed(names):
                archive.writestr(f"dataset/{name}", self.buffer.getvalue())
            archive.writestr("dataset/labels.csv", csv_text(["imagen", "light", "plate1"], rows))
        return rows

    def run_release(self, **kwargs):
        return prepare_release(
            self.archive, self.output,
            face_detectors=kwargs.pop("face_detectors", {"yunet": Detector(), "scrfd": Detector()}),
            plate_detector=kwargs.pop("plate_detector", Detector()),
            show_progress=False, **kwargs,
        )

    def test_numbering_join_keeps_all_images_and_preserves_companion_metadata(self):
        rows = self.make_archive()
        for row in rows:
            row.update(distance1="Medio", angle="Normal")
        companion = self.root / "labels.csv"
        companion.write_text(csv_text(list(rows[0]), list(reversed(rows))), encoding="utf-8")
        original_archive = self.archive.read_bytes()
        original_csv = companion.read_bytes()

        summary = self.run_release()

        self.assertEqual(summary["processed_images"], 3)
        self.assertEqual({path.name for path in self.output.iterdir()}, {"img1.png", "img2.png", "img3.png", "labels.csv"})
        with (self.output / "labels.csv").open(newline="") as stream:
            output_rows = list(csv.DictReader(stream))
        self.assertEqual([row["imagen"] for row in output_rows], ["img2.png", "img3.png"])
        self.assertTrue(all(row["distance1"] == "Medio" and row["plate1"] == "1234ABC" for row in output_rows))
        self.assertEqual(self.archive.read_bytes(), original_archive)
        self.assertEqual(companion.read_bytes(), original_csv)
        self.assertEqual((self.output / "img1.png").read_bytes(), self.buffer.getvalue())

    def test_plate_false_positive_is_excluded_and_padded_blur_restores_plate_pixels(self):
        self.make_archive(names=("frame.png",), labeled=("frame.png",))
        plate = (28, 32, 52, 48)
        false_face = Detection(plate, {"yunet": 0.99})
        # Its padded region reaches into the plate, but the original face box
        # covers less than 25% of the plate. It must still be blurred.
        real_face = Detection((18, 25, 30, 40), {"yunet": 0.9})
        summary = self.run_release(
            face_detectors={"yunet": Detector([false_face, real_face]), "scrfd": Detector()},
            plate_detector=Detector([Detection(plate, {"plate": 0.9})]),
            review=True,
        )
        self.assertEqual(summary["detections_excluded_for_plate_overlap"], 1)
        self.assertEqual(summary["face_regions_blurred"], 1)
        with Image.open(self.output / "img1.png") as image:
            result = np.array(image)
        np.testing.assert_array_equal(result[32:48, 28:52], self.pixels[32:48, 28:52])
        self.assertFalse(np.array_equal(result[25:32, 18:27], self.pixels[25:32, 18:27]))
        self.assertTrue((self.root / "public-audit/review/img1.jpg").exists())

    def test_inference_failure_does_not_publish_originals_or_final_csv(self):
        self.make_archive()
        with redirect_stderr(io.StringIO()), self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.run_release(face_detectors={"yunet": Detector(), "scrfd": Detector(error="inference failed")})
        self.assertFalse(self.output.exists())
        partial = self.root / "public-partial"
        self.assertFalse((partial / "source-image").exists())
        self.assertFalse((partial / "data/img1.png").exists())
        summary = json.loads((partial / "audit/summary.json").read_text())
        self.assertEqual(summary["status"], "failed")

    def test_conflicting_companion_csv_is_rejected_before_writing(self):
        rows = self.make_archive()
        rows[0]["plate1"] = "9999XYZ"
        (self.root / "labels.csv").write_text(csv_text(list(rows[0]), rows))
        with self.assertRaisesRegex(ValueError, "changes archived value"):
            self.run_release()
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root / "public-partial").exists())

    def test_preview_csv_references_only_preview_images(self):
        self.make_archive()
        summary = self.run_release(limit=2)
        self.assertEqual(summary["status"], "preview_complete")
        with (self.output / "labels.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["imagen"] for row in rows], ["img2.png"])
        self.assertFalse((self.output / "img3.png").exists())

    def test_duplicate_basename_and_path_traversal_are_rejected(self):
        for extra in ("other/a.png", "../escape.png"):
            self.make_archive()
            with ZipFile(self.archive, "a") as archive:
                archive.writestr(extra, self.buffer.getvalue())
            with ZipFile(self.archive) as archive, self.assertRaises(ValueError):
                build_plan(archive)

    def test_existing_output_is_not_overwritten(self):
        self.make_archive()
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_release()
        self.assertEqual(marker.read_text(), "keep")

    def test_tiles_map_faces_back_to_original_coordinates(self):
        image = Image.new("RGB", (160, 100))
        yunet = Detector([Detection((8, 8, 16, 16), {"yunet": 0.9})])
        _, faces, _, _ = detect_image(image, {"yunet": yunet, "scrfd": Detector()}, Detector(), 64, 0.25, include_tiles=True)
        self.assertIn((104.0, 44.0, 112.0, 52.0), [face.box for face in faces])
        self.assertTrue(all(max(size) <= 64 for size in yunet.sizes))

    def test_overview_rescales_once_and_maps_detections_to_source_pixels(self):
        image = Image.new("RGB", (200, 100))
        yunet = Detector([Detection((16, 8, 32, 24), {"yunet": 0.9})])
        scrfd, plate = Detector(), Detector()
        _, faces, _, _ = detect_image(image, {"yunet": yunet, "scrfd": scrfd}, plate, 64, 0.25)
        self.assertEqual([face.box for face in faces], [(50.0, 25.0, 100.0, 75.0)])
        self.assertEqual((len(yunet.sizes), len(scrfd.sizes), len(plate.sizes)), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
