import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from face_anonymisation.anonymise_faces import run_pipeline
from face_anonymisation.detectors import Detection, intersection_over_union, merge_detections
from face_anonymisation.image_processing import (
    blur_detections,
    expand_box,
    gaussian_kernel_size,
    load_image,
    save_image,
)


class FakeDetector:
    def __init__(self, detections=None, error=None):
        self.detections = list(detections or [])
        self.error = error

    def detect(self, _image):
        if self.error:
            raise RuntimeError(self.error)
        return list(self.detections)


def write_test_image(path, pattern=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if pattern:
        yy, xx = np.indices((96, 128))
        values = ((xx + yy) % 2 * 255).astype(np.uint8)
        pixels = np.stack((values, values, values), axis=-1)
        Image.fromarray(pixels).save(str(path), format="PNG")
    else:
        Image.new("RGB", (128, 96), (120, 100, 80)).save(str(path), format="PNG")


class BoxAndBlurTests(unittest.TestCase):
    def test_expansion_clips_at_image_boundaries(self):
        self.assertEqual(expand_box((2, 3, 12, 13), 0.3, (20, 15)), (0, 0, 15, 15))

    def test_iou_and_cross_detector_merge_preserve_distinct_neighbour(self):
        self.assertAlmostEqual(intersection_over_union((0, 0, 10, 10), (5, 5, 15, 15)), 1 / 7)
        items = [
            Detection((0, 0, 10, 10), {"yunet": 0.8}),
            Detection((1, 1, 11, 11), {"scrfd": 0.9}),
            Detection((14, 0, 22, 10), {"yunet": 0.7}),
        ]
        merged = merge_detections(items, 0.5)
        self.assertEqual(len(merged), 2)
        self.assertEqual(set(merged[0].sources), {"yunet", "scrfd"})
        self.assertEqual(merged[1].sources, ("yunet",))

    def test_odd_adaptive_kernels_and_synthetic_gaussian_blur(self):
        self.assertEqual(gaussian_kernel_size((1, 2), 0.5), (3, 3))
        yy, xx = np.indices((96, 128))
        checker = ((xx + yy) % 2 * 255).astype(np.uint8)
        pixels = np.stack((checker, checker, checker), axis=-1)
        source = Image.fromarray(pixels)
        detection = Detection((40, 25, 88, 73), {"yunet": 0.9})
        result, regions = blur_detections(source, [detection], padding=0.25, blur_strength=0.5)
        self.assertEqual(result.size, source.size)
        self.assertEqual(len(regions), 1)
        self.assertFalse(np.array_equal(np.asarray(result.crop(regions[0])), np.asarray(source.crop(regions[0]))))
        self.assertEqual(result.getpixel((0, 0)), source.getpixel((0, 0)))

    def test_anonymised_image_save_preserves_dimensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "image.png"
            source = Image.new("RGB", (47, 31), (20, 40, 60))
            asset = type("Asset", (), {"image": source, "image_format": "PNG", "metadata": {}})()
            save_image(asset, path)
            self.assertEqual(load_image(path).image.size, (47, 31))


class DatasetPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "BLPR-A"
        self.output = self.root / "BLPR-A-anonymised"

    def tearDown(self):
        self.temp.cleanup()

    def make_image(self, relative="train/images/one.png", pattern=False):
        path = self.source / relative
        write_test_image(path, pattern=pattern)
        return path

    def run_fake(self, detector="yunet", **kwargs):
        instances = kwargs.pop("detector_instances", {detector: FakeDetector()})
        return run_pipeline(
            self.source,
            self.output,
            detector_mode=detector,
            detector_instances=instances,
            show_progress=False,
            **kwargs
        )

    def test_directory_structure_and_label_contents_are_preserved(self):
        image_path = self.make_image()
        label = self.source / "train/labels/one.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("1 0.5 0.5 0.2 0.1\n", encoding="utf-8")
        yaml = self.source / "data.yaml"
        yaml.write_text("train: ../train/images\n", encoding="utf-8")
        face = Detection((42, 22, 82, 64), {"yunet": 0.92})

        summary = self.run_fake(detector_instances={"yunet": FakeDetector([face])})

        output_image = self.output / "train/images/one.png"
        self.assertTrue(output_image.is_file())
        with Image.open(str(output_image)) as output_pixels, Image.open(str(image_path)) as source_pixels:
            self.assertEqual(output_pixels.size, source_pixels.size)
        self.assertEqual((self.output / "train/labels/one.txt").read_bytes(), label.read_bytes())
        self.assertEqual((self.output / "data.yaml").read_bytes(), yaml.read_bytes())
        self.assertEqual(summary["total_anonymised_face_regions"], 1)

    def test_no_detections_copy_source_bytes_and_are_not_called_face_free(self):
        original = self.make_image()
        source_bytes = original.read_bytes()
        summary = self.run_fake()
        self.assertEqual((self.output / "train/images/one.png").read_bytes(), source_bytes)
        self.assertEqual(summary["images_without_detected_faces"], 1)
        self.assertNotIn("verified_face_free_images", summary)

    def test_multiple_faces_are_blurred_and_counted(self):
        self.make_image(pattern=True)
        faces = [
            Detection((15, 15, 45, 48), {"yunet": 0.8}),
            Detection((77, 30, 112, 70), {"yunet": 0.86}),
        ]
        summary = self.run_fake(detector_instances={"yunet": FakeDetector(faces)})
        self.assertEqual(summary["total_anonymised_face_regions"], 2)
        with open(self.output / "face_anonymisation_report.csv", newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(int(row["final_merged_detections"]), 2)

    def test_corrupt_image_is_reported_and_not_saved_as_success(self):
        path = self.source / "test/images/broken.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not an image")
        summary = self.run_fake()
        self.assertEqual(summary["images_with_processing_failures"], 1)
        self.assertFalse((self.output / "test/images/broken.jpg").exists())
        report = json.loads((self.output / "face_anonymisation_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["processing_status_counts"]["failed_read"], 1)

    def test_existing_image_output_is_protected_without_overwrite(self):
        self.make_image()
        self.run_fake()
        target = self.output / "train/images/one.png"
        target.write_bytes(b"keep me")
        second = self.run_fake(detector_instances={"yunet": FakeDetector([
            Detection((40, 20, 80, 65), {"yunet": 0.95})
        ])})
        self.assertEqual(target.read_bytes(), b"keep me")
        self.assertEqual(second["images_skipped_because_outputs_exist"], 1)
        third = self.run_fake(overwrite=True, detector_instances={"yunet": FakeDetector([
            Detection((40, 20, 80, 65), {"yunet": 0.95})
        ])})
        self.assertEqual(third["total_anonymised_face_regions"], 1)
        self.assertNotEqual(target.read_bytes(), b"keep me")

    def test_combined_mode_retains_faces_found_by_only_one_model(self):
        self.make_image("train/images/yunet_only.png")
        yunet_face = Detection((40, 20, 80, 65), {"yunet": 0.88})
        scrfd_face = Detection((42, 22, 84, 68), {"scrfd": 0.91})
        summary = self.run_fake(
            detector="both",
            detector_instances={
                "yunet": FakeDetector([yunet_face]),
                "scrfd": FakeDetector([]),
            },
        )
        self.assertEqual(summary["total_anonymised_face_regions"], 1)
        self.assertEqual(summary["faces_detected_exclusively_by_yunet"], 1)
        self.assertEqual(summary["faces_detected_exclusively_by_scrfd"], 0)

        other_output = self.root / "other-output"
        summary = run_pipeline(
            self.source,
            other_output,
            detector_mode="both",
            detector_instances={
                "yunet": FakeDetector([]),
                "scrfd": FakeDetector([scrfd_face]),
            },
            show_progress=False,
        )
        self.assertEqual(summary["faces_detected_exclusively_by_scrfd"], 1)

    def test_reports_have_one_row_per_discovered_image_and_consistent_counts(self):
        self.make_image("train/images/face.png")
        self.make_image("valid/images/clear.png")
        face = Detection((20, 15, 60, 60), {"yunet": 0.75})
        summary = self.run_fake(detector_instances={"yunet": FakeDetector([face])})
        with open(self.output / "face_anonymisation_report.csv", newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), summary["total_images_discovered"])
        self.assertEqual(
            summary["successfully_processed_images"] + summary["images_with_processing_failures"]
            + summary["images_skipped_because_outputs_exist"],
            summary["images_selected_for_processing"],
        )
        self.assertTrue((self.output / "face_anonymisation_summary.json").is_file())

    def test_limit_and_dry_run_write_reports_but_no_image_output(self):
        self.make_image("train/images/one.png")
        self.make_image("valid/images/two.png")
        summary = self.run_fake(limit=1, dry_run=True)
        self.assertEqual(summary["total_images_discovered"], 2)
        self.assertEqual(summary["images_selected_for_processing"], 1)
        self.assertFalse((self.output / "train/images/one.png").exists())

    def test_inference_failure_does_not_save_original_as_anonymised(self):
        self.make_image()
        summary = self.run_fake(detector_instances={"yunet": FakeDetector(error="model failed")})
        self.assertEqual(summary["images_with_processing_failures"], 1)
        self.assertFalse((self.output / "train/images/one.png").exists())


class OptionalModelIntegrationTest(unittest.TestCase):
    models = Path(__file__).resolve().parents[1] / "models"
    yunet_path = models / "face_detection_yunet_2023mar.onnx"
    scrfd_path = models / "scrfd_2.5g.onnx"
    dependencies_available = True
    try:
        import cv2  # noqa: F401
        import insightface  # noqa: F401
    except ImportError:
        dependencies_available = False

    @unittest.skipUnless(
        dependencies_available and yunet_path.is_file() and scrfd_path.is_file(),
        "Install the optional YuNet and SCRFD model files to run integration inference",
    )
    def test_both_official_detectors_load_and_run_cpu_inference(self):
        from face_anonymisation.detectors import create_detectors

        detectors = create_detectors("both", self.yunet_path, self.scrfd_path, 0.5, 0.4)
        blank = np.zeros((128, 128, 3), dtype=np.uint8)
        self.assertIsInstance(detectors["yunet"].detect(blank), list)
        self.assertIsInstance(detectors["scrfd"].detect(blank), list)


if __name__ == "__main__":
    unittest.main()
