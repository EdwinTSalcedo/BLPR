"""Face detector adapters and cross-model detection matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """A face box in pixel ``(x1, y1, x2, y2)`` coordinates and model scores."""

    box: Box
    scores: Mapping[str, float]

    @property
    def sources(self) -> Tuple[str, ...]:
        return tuple(sorted(self.scores))


def intersection_over_union(first: Box, second: Box) -> float:
    """Return IoU for two unexpanded xyxy boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def merge_detections(detections: Iterable[Detection], iou_threshold: float) -> List[Detection]:
    """Merge matching YuNet and SCRFD boxes, keeping unmatched boxes."""
    items = list(detections)
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be between 0 and 1")

    pairs = []
    for first_index, first in enumerate(items):
        for second_index in range(first_index + 1, len(items)):
            second = items[second_index]
            if first.sources == second.sources:
                continue
            if len(first.sources) != 1 or len(second.sources) != 1:
                continue
            overlap = intersection_over_union(first.box, second.box)
            if overlap > 0.0 and overlap >= iou_threshold:
                pairs.append((overlap, first_index, second_index))
    pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))

    matched = set()
    output = []
    for _overlap, first_index, second_index in pairs:
        if first_index in matched or second_index in matched:
            continue
        first, second = items[first_index], items[second_index]
        scores = dict(first.scores)
        scores.update(second.scores)
        box = (
            min(first.box[0], second.box[0]),
            min(first.box[1], second.box[1]),
            max(first.box[2], second.box[2]),
            max(first.box[3], second.box[3]),
        )
        output.append(Detection(box=box, scores=scores))
        matched.update((first_index, second_index))

    output.extend(item for index, item in enumerate(items) if index not in matched)
    output.sort(key=lambda item: (item.box[1], item.box[0], item.box[3], item.box[2]))
    return output


class YuNetDetector:
    """OpenCV FaceDetectorYN adapter running on the CPU."""

    name = "yunet"

    def __init__(self, model_path: Path, confidence_threshold: float) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("YuNet needs opencv-python-headless; install requirements.txt") from exc
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("This OpenCV build does not provide cv2.FaceDetectorYN")
        if not model_path.is_file():
            raise FileNotFoundError("YuNet ONNX model not found: {}".format(model_path))
        self._cv2 = cv2
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(model_path), "", (320, 320), confidence_threshold, 0.3, 5000,
                cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
            )
        except Exception as exc:
            raise RuntimeError("Could not load YuNet model {}: {}".format(model_path, exc)) from exc

    def detect(self, image_bgr) -> List[Detection]:
        height, width = image_bgr.shape[:2]
        self._detector.setInputSize((width, height))
        _status, rows = self._detector.detect(image_bgr)
        if rows is None:
            return []
        detections = []
        for row in rows:
            x, y, box_width, box_height = (float(value) for value in row[:4])
            detections.append(Detection(
                (x, y, x + box_width, y + box_height),
                {self.name: float(row[14])},
            ))
        return detections


class SCRFDDetector:
    """InsightFace's SCRFD ONNX detection model, without recognition models."""

    name = "scrfd"

    def __init__(self, model_path: Path, confidence_threshold: float) -> None:
        if not model_path.is_file():
            raise FileNotFoundError("SCRFD ONNX model not found: {}".format(model_path))
        try:
            import insightface
        except ImportError as exc:
            raise RuntimeError("SCRFD needs insightface; install requirements.txt") from exc
        try:
            # Select CPU before ONNX Runtime creates the session.
            self._detector = insightface.model_zoo.get_model(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            self._detector.prepare(ctx_id=-1, det_thresh=confidence_threshold)
        except Exception as exc:
            raise RuntimeError("Could not load SCRFD model {}: {}".format(model_path, exc)) from exc
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

    def detect(self, image_bgr) -> List[Detection]:
        try:
            rows, _keypoints = self._detector.detect(image_bgr, max_num=0)
        except Exception as exc:
            raise RuntimeError("SCRFD inference failed: {}".format(exc)) from exc
        if rows is None:
            return []
        return [
            Detection(
                tuple(float(value) for value in row[:4]),
                {self.name: float(row[4])},
            )
            for row in rows
        ]


def create_detectors(
    mode: str,
    yunet_model: Path,
    scrfd_model: Path,
    yunet_threshold: float,
    scrfd_threshold: float,
) -> Dict[str, object]:
    """Load requested detector backends; model loading never downloads weights."""
    result = {}
    if mode in ("yunet", "both"):
        result["yunet"] = YuNetDetector(yunet_model, yunet_threshold)
    if mode in ("scrfd", "both"):
        result["scrfd"] = SCRFDDetector(scrfd_model, scrfd_threshold)
    return result
