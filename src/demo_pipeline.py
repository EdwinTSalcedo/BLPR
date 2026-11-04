"""
ALPR Video Demo
---------------

"""


import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
DEFAULT_PLATE_MODEL = MODELS_DIR / "best_lpd.pt"
DEFAULT_CHAR_MODEL = MODELS_DIR / "best_lpr.pt"
DEFAULT_VIDEO_INPUT = REPO_ROOT / "data" / "videos" / "video.mp4"
DEFAULT_VIDEO_OUTPUT = REPO_ROOT / "data" / "inference" / "videos" / "video.mp4"


try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False

    class _FallbackLogger:
        def _log(self, level: str, message: str, *args, **kwargs):
            if args or kwargs:
                try:
                    message = message.format(*args, **kwargs)
                except Exception:
                    pass
            print(f"[{level}] {message}")

        def debug(self, message: str, *args, **kwargs):
            self._log("DEBUG", message, *args, **kwargs)

        def info(self, message: str, *args, **kwargs):
            self._log("INFO", message, *args, **kwargs)

        def success(self, message: str, *args, **kwargs):
            self._log("SUCCESS", message, *args, **kwargs)

        def warning(self, message: str, *args, **kwargs):
            self._log("WARN", message, *args, **kwargs)

        def error(self, message: str, *args, **kwargs):
            self._log("ERROR", message, *args, **kwargs)

        def remove(self):
            return None

        def add(self, *args, **kwargs):
            return None

    logger = _FallbackLogger()

try:
    import supervision as sv
    SV_AVAILABLE = True
except ImportError:
    SV_AVAILABLE = False
    logger.error("supervision not installed.")
    logger.error('Run: python -m pip install "supervision>=0.25.1,<0.31"')
    raise SystemExit(1)

if not hasattr(sv, "ByteTrack"):
    logger.error("This video pipeline requires supervision's ByteTrack API (removed in 0.31).")
    logger.error('Run: python -m pip install "supervision>=0.25.1,<0.31"')
    raise SystemExit(1)

try:
    from agno.agent import Agent
    from agno.models.ollama import Ollama
    from agno.media import Image as AgnoImage
    AGNO_AVAILABLE = True
except ImportError:
    AGNO_AVAILABLE = False


IGNORED_STRINGS = {"BOLIVIA", "_"}
LLM_GEMMA_ID    = "gemma3:4b"

COLOR_VEHICLE   = (0,   200,   0) 
COLOR_PLATE     = (255, 140,   0) 
COLOR_TEXT_OK   = (0,   255,   0)   
COLOR_TEXT_WEAK = (0,   165, 255)  
COLOR_TEXT_VLM  = (255,  50, 255)  
COLOR_TEXT_LOCK = (0,   255, 255)   
FONT            = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE      = 0.75
THICKNESS       = 2

PLATE_CLASSES   = {"plate", "placa", "license plate", "licence plate"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "motorbike",
                   "vehicle", "auto", "vehiculo"}

MIN_VOTES_TO_LOCK = 3
TRACK_BUFFER_SECONDS = 2.0
TRACK_MATCHING_THRESHOLD = 0.9
SUSPECT_DEFAULT_TEXTS = {"ABC1234"}

DEBUG_WINDOW_NAME = "ALPR - Inference Debug"
DEBUG_TILE_SIZE = (420, 280)


def configure_logger(debug_enabled: bool):
    level = "DEBUG" if debug_enabled else "INFO"
    if LOGURU_AVAILABLE:
        logger.remove()
        logger.add(
            sys.stderr,
            level=level,
            colorize=True,
            format=(
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{message}</cyan>"
            ),
        )
    elif debug_enabled:
        logger.warning("loguru is not installed; falling back to plain logging.")



class VoteBuffer:
    """Stabilizes displayed OCR text and permanently locks strong consensus."""

    def __init__(self, maxlen: int = 20, min_chars: int = 6):
        self._readings: deque = deque(maxlen=maxlen)
        self._best: tuple     = ("", False)
        self._best_votes      = 0
        self._locked          = None
        self._min_chars       = min_chars
        self._dirty           = True

    def add(self, text: str, is_vlm: bool, confidence: float = 0.0):
        if not text or self._locked is not None:
            return
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        if not np.isfinite(confidence):
            confidence = 0.0
        self._readings.append((text, is_vlm, max(confidence, 0.0)))
        self._dirty = True

    def _weighted_character_consensus(self) -> str:
        length_groups = {}
        for text, is_vlm, confidence in self._readings:
            if is_vlm or confidence <= 0.0 or len(text) < self._min_chars:
                continue
            length_groups.setdefault(len(text), []).append((text, confidence))

        best_text = ""
        best_key = None
        for observations in length_groups.values():
            if len(observations) < MIN_VOTES_TO_LOCK:
                continue

            total_weight = sum(confidence for _, confidence in observations)
            consensus_chars = []
            for position in range(len(observations[0][0])):
                char_weights = {}
                for text, confidence in observations:
                    char = text[position]
                    char_weights[char] = char_weights.get(char, 0.0) + confidence
                char, weight = max(char_weights.items(), key=lambda item: item[1])
                if weight <= total_weight / 2.0:
                    consensus_chars = []
                    break
                consensus_chars.append(char)

            consensus = "".join(consensus_chars)
            if not consensus or not any(text == consensus for text, _ in observations):
                continue

            candidate_key = (len(observations), total_weight)
            if best_key is None or candidate_key > best_key:
                best_text = consensus
                best_key = candidate_key

        return best_text

    def best(self) -> tuple:
        if self._locked is not None:
            return self._locked
        if not self._dirty:
            return self._best
        if not self._readings:
            self._best       = ("", False)
            self._best_votes = 0
            self._dirty      = False
            return self._best

        stats = {}
        for text, is_vlm, confidence in self._readings:
            if text not in stats:
                stats[text] = {"count": 0, "confidence": 0.0, "vlm": []}
            stats[text]["count"] += 1
            if not is_vlm:
                stats[text]["confidence"] += confidence
            stats[text]["vlm"].append(is_vlm)

        winner = self._best[0] if self._best[0] in stats else next(iter(stats))
        winner_key = (stats[winner]["count"], stats[winner]["confidence"])
        for text, values in stats.items():
            candidate_key = (values["count"], values["confidence"])
            if candidate_key > winner_key:
                winner = text
                winner_key = candidate_key

        votes = stats[winner]["count"]
        vlm_flags = stats[winner]["vlm"]
        self._best       = (winner, all(vlm_flags))
        self._best_votes = votes

        if votes >= MIN_VOTES_TO_LOCK:
            self._locked = self._best
        else:
            weighted_text = self._weighted_character_consensus()
            if weighted_text:
                self._best = (weighted_text, False)
                self._best_votes = stats[weighted_text]["count"]
                self._locked = self._best

        self._dirty = False
        return self._best

    def count(self) -> int:
        return len(self._readings)

    def best_count(self) -> int:
        if self._dirty:
            self.best()
        return self._best_votes

    def is_locked(self) -> bool:
        if self._dirty:
            self.best()
        return self._locked is not None


def create_plate_tracker(frame_rate: float):
    """Create a ByteTrack instance tuned for plate-sized detections."""
    tracker_frame_rate = max(1, int(round(frame_rate)))
    lost_track_buffer = max(1, int(round(30 * TRACK_BUFFER_SECONDS)))

    try:
        return sv.ByteTrack(
            track_activation_threshold=0.1,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=TRACK_MATCHING_THRESHOLD,
            frame_rate=tracker_frame_rate,
            minimum_consecutive_frames=1,
        )
    except TypeError:
        try:
            return sv.ByteTrack(
                track_activation_threshold=0.1,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=TRACK_MATCHING_THRESHOLD,
                frame_rate=tracker_frame_rate,
            )
        except TypeError:
            try:
                return sv.ByteTrack(
                    track_thresh=0.1,
                    track_buffer=lost_track_buffer,
                    match_thresh=TRACK_MATCHING_THRESHOLD,
                    frame_rate=tracker_frame_rate,
                )
            except TypeError:
                return sv.ByteTrack()


def coerce_track_id(tracker_ids, index: int):
    if tracker_ids is None:
        return None
    raw_id = tracker_ids[index]
    if raw_id is None:
        return None
    try:
        track_id = int(raw_id)
    except (TypeError, ValueError):
        return None
    return track_id if track_id >= 0 else None


def prune_stale_vote_buffers(vote_buffers: dict, track_last_seen: dict,
                             frame_idx: int, ttl_frames: int,
                             track_best_inferences=None):
    stale_track_ids = [
        track_id for track_id, last_seen in track_last_seen.items()
        if frame_idx - last_seen > ttl_frames
    ]
    for track_id in stale_track_ids:
        vote_buffers.pop(track_id, None)
        track_last_seen.pop(track_id, None)
        if track_best_inferences is not None:
            track_best_inferences.pop(track_id, None)


# VLM FALLBACK
class LLMPlateReader:
    _DESC  = "You are a precise License Plate Recognition system."
    _INSTR = [
        "Return ONLY the alphanumeric characters seen on the plate.",
        "Do not include country names like BOLIVIA.",
        "No markdown, no explanations, no spaces.",
    ]

    def __init__(self):
        self.agent = None
        if not AGNO_AVAILABLE:
            logger.warning("agno not installed; VLM fallback disabled.")
            return
        logger.info("Initialising Gemma3 VLM fallback agent.")
        self.agent = Agent(
            model=Ollama(id=LLM_GEMMA_ID),
            description=self._DESC,
            instructions=self._INSTR,
        )

    def infer(self, image: np.ndarray) -> str:
        if self.agent is None or image is None or image.size == 0:
            return ""
        try:
            ok, buf = cv2.imencode(".jpg", image)
            if not ok:
                return ""
            resp = self.agent.run(
                "Read this license plate text.",
                images=[AgnoImage(content=buf.tobytes())],
            )
            raw = resp.content.strip().upper()
            return "".join(c for c in raw if c.isalnum())
        except Exception as exc:
            logger.warning("VLM error: {}", exc)
            return ""


def context_heur_ad_gamma(image,
                           contrast_threshold=60,
                           mean_v_range=(80, 160),
                           gamma_bounds=(0.6, 1.5)):
    hsv           = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v       = cv2.split(hsv)
    mean_v        = float(np.mean(v))
    std_v         = float(np.std(v))
    if std_v > contrast_threshold:
        return image
    if mean_v_range[0] <= mean_v <= mean_v_range[1]:
        return image
    if mean_v < 1:
        return image
    gamma     = np.log(128.0 / 255.0) / np.log(mean_v / 255.0)
    gamma     = float(np.clip(gamma, gamma_bounds[0], gamma_bounds[1]))
    inv_gamma = 1.0 / gamma
    table     = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype="uint8"
    )
    return cv2.cvtColor(cv2.merge([h, s, cv2.LUT(v, table)]),
                        cv2.COLOR_HSV2BGR)


def illumination_clahe(image):
    lab     = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def apply_nlm_denoise(image):
    return cv2.fastNlMeansDenoisingColored(
        image, None, h=10, hColor=10,
        templateWindowSize=7, searchWindowSize=21,
    )


def order_points(pts):
    rect    = np.zeros((4, 2), dtype="float32")
    s       = pts.sum(axis=1)
    d       = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def apply_heur_3d_morph(roi):
    h, w    = roi.shape[:2]
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    block   = max(3, int(h * 0.35) | 1)
    binary  = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, 10,
    )
    mask    = np.zeros_like(binary)
    mask[int(h * 0.20):int(h * 0.90), int(w * 0.05):int(w * 0.95)] = 255
    binary  = cv2.bitwise_and(binary, mask)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return roi, False
    lc   = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(lc)
    if not (0.10 * (w * h) <= area <= 0.60 * (w * h)):
        return roi, False
    hull      = cv2.convexHull(lc)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0 or float(area) / hull_area < 0.45:
        return roi, False
    rect = cv2.minAreaRect(lc)
    box  = np.int32(cv2.boxPoints(rect))
    ops  = order_points(box.astype("float32"))
    bx, by, bw, bh = cv2.boundingRect(ops)
    dst = np.array(
        [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(ops, dst)
    return cv2.warpPerspective(
        roi, M, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    ), True


def smart_perspective_router(roi,
                              foreshortening_threshold=1.05,
                              angle_threshold=12.5,
                              width_threshold=0.20):
    h, w      = roi.shape[:2]
    scout     = illumination_clahe(roi)
    gray      = cv2.cvtColor(scout, cv2.COLOR_BGR2GRAY)
    blurred   = cv2.GaussianBlur(gray, (5, 5), 0)
    edges     = cv2.Canny(blurred, 30, 150)
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    closed    = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return roi, False
    lc = max(contours, key=cv2.contourArea)
    if cv2.contourArea(lc) < (w * h * 0.15):
        return roi, False
    epsilon = 0.05 * cv2.arcLength(lc, True)
    approx  = cv2.approxPolyDP(lc, epsilon, True)
    if len(approx) == 4:
        ops      = order_points(np.float32(approx.reshape(4, 2)))
        wA, wB   = np.linalg.norm(ops[2] - ops[3]), np.linalg.norm(ops[1] - ops[0])
        hA, hB   = np.linalg.norm(ops[1] - ops[2]), np.linalg.norm(ops[0] - ops[3])
        minW, maxW = min(wA, wB), max(wA, wB)
        minH, maxH = min(hA, hB), max(hA, hB)
        if minW == 0 or minH == 0:
            return roi, False
        max_fore = max(maxW / minW, maxH / minH)
        dx, dy   = ops[1][0] - ops[0][0], ops[1][1] - ops[0][1]
        angle    = abs(np.degrees(np.arctan2(dy, dx)))
        if max_fore > foreshortening_threshold or angle > angle_threshold:
            bx, by, bw, bh = cv2.boundingRect(ops)
            dst = np.array(
                [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]],
                dtype="float32",
            )
            if np.max(np.linalg.norm(ops - dst, axis=1)) > (w * width_threshold):
                return roi, False
            M = cv2.getPerspectiveTransform(ops, dst)
            return cv2.warpPerspective(
                roi, M, (w, h), flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            ), True
        elif max_fore < 1.06 and angle < 5.0:
            morphed, ok = apply_heur_3d_morph(apply_nlm_denoise(roi))
            return morphed, ok
    return roi, False


def preprocess_roi_steps(roi: np.ndarray):
    roi_routed, _ = smart_perspective_router(roi.copy())
    roi_processed = context_heur_ad_gamma(roi_routed.copy())
    return roi_routed, roi_processed


def preprocess_roi(roi: np.ndarray) -> np.ndarray:
    _, roi_processed = preprocess_roi_steps(roi)
    return roi_processed



def run_ocr(roi_img: np.ndarray, char_model, conf_threshold: float,
            return_debug: bool = False):
    """Returns (text, min_conf, char_count), plus debug details if requested."""
    resized = cv2.resize(roi_img, (1020, 730), interpolation=cv2.INTER_CUBIC)
    results = char_model.predict(resized, verbose=False, conf=conf_threshold)

    chars = []
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id   = int(box.cls[0].item())
            char_str = char_model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].item())
            ignored = char_str.upper() in IGNORED_STRINGS or "_" in char_str
            detections.append({
                "char":    char_str,
                "bbox":    (float(x1), float(y1), float(x2), float(y2)),
                "conf":    conf,
                "ignored": ignored,
            })
            if ignored:
                continue
            chars.append({
                "char": char_str,
                "cx":   (x1 + x2) / 2,
                "cy":   (y1 + y2) / 2,
                "h":    y2 - y1,
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "conf": conf,
            })

    if not chars:
        if return_debug:
            return "", 0.0, 0, {
                "detections": detections,
                "ordered_chars": [],
                "resized_shape": resized.shape[:2],
            }
        return "", 0.0, 0

    chars.sort(key=lambda c: c["cy"])
    lines, current = [], [chars[0]]
    for c in chars[1:]:
        if (c["cy"] - current[-1]["cy"]) > (current[-1]["h"] * 0.5):
            lines.append(current)
            current = [c]
        else:
            current.append(c)
    lines.append(current)

    text, confs, ordered_chars = "", [], []
    for line in lines:
        line.sort(key=lambda c: c["cx"])
        for c in line:
            text += c["char"]
            confs.append(c["conf"])
            ordered_chars.append(c)

    if return_debug:
        return text, min(confs) if confs else 0.0, len(text), {
            "detections": detections,
            "ordered_chars": ordered_chars,
            "resized_shape": resized.shape[:2],
        }
    return text, min(confs) if confs else 0.0, len(text)


def draw_labeled_box(frame, x1, y1, x2, y2, label, box_color, text_color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, THICKNESS)
    if not label:
        return
    (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
    pad    = 4
    bg_y1  = max(y1 - th - bl - pad * 2, 0)
    bg_y2  = y1
    cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + pad * 2, bg_y2),
                  box_color, -1)
    cv2.putText(frame, label, (x1 + pad, bg_y2 - bl - 1),
                FONT, FONT_SCALE, text_color, THICKNESS, cv2.LINE_AA)


def show_preview(window_name: str, frame: np.ndarray, scale: float) -> bool:
    """
    Displays frame in a named window.
    Returns False if the user pressed Q (quit signal).
    """
    if scale != 1.0:
        h, w = frame.shape[:2]
        preview = cv2.resize(frame, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    else:
        preview = frame

    cv2.imshow(window_name, preview)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), ord("Q"), 27)  # 27 = ESC


def parse_suspect_texts(raw_texts: str) -> set[str]:
    if raw_texts is None:
        return set(SUSPECT_DEFAULT_TEXTS)
    return {t.strip().upper() for t in raw_texts.split(",") if t.strip()}


def is_suspect_text(text: str, suspect_texts: set[str]) -> bool:
    return bool(text) and text.upper() in suspect_texts


def get_detection_confidence(detections, index: int) -> float:
    confidences = getattr(detections, "confidence", None)
    if confidences is None or index >= len(confidences):
        return 0.0
    try:
        return float(confidences[index])
    except (TypeError, ValueError):
        return 0.0


def inference_score(text: str, min_conf: float, plate_conf: float,
                    is_vlm: bool, min_chars: int,
                    suspect_texts: set[str]) -> float:
    if not text:
        return 0.0
    score = min(len(text), min_chars + 2) * 10.0
    score += max(min_conf, 0.0) * 20.0
    score += max(plate_conf, 0.0) * 10.0
    if len(text) >= min_chars:
        score += 30.0
    if is_vlm:
        score -= 5.0
    if is_suspect_text(text, suspect_texts):
        score -= 50.0
    return score


def should_run_track_inference(track_best: dict, track_id: int,
                               plate_conf: float, args,
                               suspect_texts: set[str]):
    if not args.debug or args.debug_no_track_cache:
        return True, "track cache disabled"
    best = track_best.get(track_id)
    if best is None:
        return True, "new track"
    if not best.get("text"):
        return True, "cached inference is empty"
    if is_suspect_text(best.get("text", ""), suspect_texts):
        return True, "cached text is suspect"
    if best.get("length", 0) < args.min_chars:
        return True, "cached text is shorter than min chars"
    best_plate_conf = best.get("plate_conf", 0.0)
    if plate_conf >= best_plate_conf + args.debug_min_conf_delta:
        return True, "plate detector confidence improved"
    return False, "cached track inference reused"


def update_track_best(track_best: dict, track_id: int, record: dict,
                      args, suspect_texts: set[str]):
    score = inference_score(
        record.get("text", ""),
        record.get("min_conf", 0.0),
        record.get("plate_conf", 0.0),
        record.get("is_vlm", False),
        args.min_chars,
        suspect_texts,
    )
    record["score"] = score
    if not record.get("text"):
        return record, False, "empty inference"

    previous = track_best.get(track_id)
    if previous is None or score >= previous.get("score", 0.0) + args.debug_min_score_delta:
        track_best[track_id] = {
            "text":       record.get("text", ""),
            "is_vlm":     record.get("is_vlm", False),
            "length":     record.get("length", 0),
            "min_conf":   record.get("min_conf", 0.0),
            "plate_conf": record.get("plate_conf", 0.0),
            "score":      score,
            "frame_idx":  record.get("frame_idx", 0),
        }
        return record, True, "new best inference"

    record["superseded_by_cached"] = True
    record["cached_text"] = previous.get("text", "")
    record["cached_is_vlm"] = previous.get("is_vlm", False)
    record["cached_score"] = previous.get("score", 0.0)
    return record, False, "previous best kept"


def cached_inference_record(track_best: dict, track_id: int, frame_idx: int,
                            bbox: tuple, plate_conf: float, reason: str):
    best = track_best[track_id]
    return {
        "text":       best.get("text", ""),
        "is_vlm":     best.get("is_vlm", False),
        "length":     best.get("length", 0),
        "min_conf":   best.get("min_conf", 0.0),
        "plate_conf": plate_conf,
        "score":      best.get("score", 0.0),
        "frame_idx":  frame_idx,
        "bbox":       bbox,
        "infer_was_run": False,
        "skip_reason": reason,
    }


def debug_text(text: str, max_chars: int = 58) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


def draw_text_lines(image: np.ndarray, lines: list[str], x: int, y: int,
                    color=(230, 230, 230), line_height: int = 21):
    for idx, line in enumerate(lines):
        cv2.putText(
            image, debug_text(line), (x, y + idx * line_height),
            FONT, 0.52, color, 1, cv2.LINE_AA,
        )


def make_debug_tile(image, title: str, size=DEBUG_TILE_SIZE):
    tile_w, tile_h = size
    header_h = 30
    body_h = tile_h - header_h
    tile = np.full((tile_h, tile_w, 3), 20, dtype=np.uint8)
    cv2.rectangle(tile, (0, 0), (tile_w, header_h), (50, 50, 50), -1)
    cv2.putText(tile, title, (8, 21), FONT, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)

    if image is None or image.size == 0:
        draw_text_lines(tile, ["<empty>"], 10, header_h + 35,
                        color=(120, 120, 255))
        return tile

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return tile
    scale = min(tile_w / w, body_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (tile_w - new_w) // 2
    y0 = header_h + (body_h - new_h) // 2
    tile[y0:y0 + new_h, x0:x0 + new_w] = resized
    return tile


def make_debug_text_panel(lines: list[str], title: str = "Inference"):
    tile_w, tile_h = DEBUG_TILE_SIZE
    panel = np.full((tile_h, tile_w, 3), 20, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (tile_w, 30), (50, 50, 50), -1)
    cv2.putText(panel, title, (8, 21), FONT, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    draw_text_lines(panel, lines, 10, 55)
    return panel


def draw_ocr_debug_image(roi_processed: np.ndarray, ocr_debug: dict):
    if roi_processed is None or roi_processed.size == 0:
        return None
    canvas = cv2.resize(roi_processed, (1020, 730), interpolation=cv2.INTER_CUBIC)
    detections = ocr_debug.get("detections", []) if ocr_debug else []
    for det in detections:
        x1, y1, x2, y2 = map(int, det.get("bbox", (0, 0, 0, 0)))
        ignored = det.get("ignored", False)
        color = (0, 0, 255) if ignored else (0, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{det.get('char', '?')} {det.get('conf', 0.0):.2f}"
        cv2.putText(canvas, label, (x1, max(y1 - 5, 15)), FONT, 0.6,
                    color, 2, cv2.LINE_AA)
    return canvas


def build_debug_grid(frame: np.ndarray, record: dict, track_id: int):
    frame_view = frame.copy()
    x1, y1, x2, y2 = record.get("bbox", (0, 0, 0, 0))
    cv2.rectangle(frame_view, (x1, y1), (x2, y2), COLOR_PLATE, THICKNESS)
    cv2.putText(frame_view, f"track {track_id}", (x1, max(20, y1 - 8)),
                FONT, 0.65, COLOR_TEXT_LOCK, 2, cv2.LINE_AA)

    ocr_image = draw_ocr_debug_image(record.get("roi_processed"),
                                     record.get("ocr_debug", {}))
    vlm_status = "not triggered"
    if record.get("vlm_triggered"):
        vlm_status = f"{record.get('vlm_text', '')} ({record.get('vlm_reason', '')})"

    lines = [
        f"frame: {record.get('frame_idx')}  track: {track_id}",
        f"bbox: {record.get('bbox')}",
        f"LPD conf: {record.get('plate_conf', 0.0):.3f}",
        f"OCR: {record.get('ocr_text', record.get('text', ''))}",
        f"OCR len/min: {record.get('ocr_length', 0)}/{record.get('min_chars', 0)}",
        f"OCR min conf: {record.get('min_conf', 0.0):.3f}",
        f"VLM: {vlm_status}",
        f"effective: {record.get('text', '')}  vlm={record.get('is_vlm', False)}",
        f"score: {record.get('score', 0.0):.2f}",
        f"decision: {record.get('decision', '')}",
        f"pause: SPACE/ENTER next, Q/ESC quit",
    ]
    if record.get("superseded_by_cached"):
        lines.insert(9, f"cached kept: {record.get('cached_text', '')}")
    if record.get("suspect"):
        lines.insert(9, "WARNING: suspect default-like text")

    tiles = [
        make_debug_tile(frame_view, "Frame + Track"),
        make_debug_tile(record.get("roi"), "Extracted ROI"),
        make_debug_tile(record.get("roi_routed"), "Preprocessed ROI"),
        make_debug_tile(record.get("roi_processed"), "Processed ROI"),
        make_debug_tile(ocr_image, "OCR Boxes + Conf"),
        make_debug_text_panel(lines),
    ]
    top = np.hstack(tiles[:3])
    bottom = np.hstack(tiles[3:])
    return np.vstack([top, bottom])


def show_debug_inference(frame: np.ndarray, record: dict, track_id: int, args):
    grid = build_debug_grid(frame, record, track_id)
    if args.debug_window_scale != 1.0:
        h, w = grid.shape[:2]
        grid = cv2.resize(
            grid,
            (max(1, int(w * args.debug_window_scale)),
             max(1, int(h * args.debug_window_scale))),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imshow(DEBUG_WINDOW_NAME, grid)

    if args.debug_save_dir:
        os.makedirs(args.debug_save_dir, exist_ok=True)
        out_name = f"frame_{record.get('frame_idx'):06d}_track_{track_id}.jpg"
        cv2.imwrite(os.path.join(args.debug_save_dir, out_name), grid)

    if args.debug_no_pause:
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), ord("Q"), 27)

    logger.info("Paused after inference. Press SPACE/ENTER for next, Q/ESC to quit.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return False
        if key in (ord(" "), 13, 10, ord("n"), ord("N")):
            return True

def process_video(args):
    configure_logger(args.debug)
    suspect_texts = parse_suspect_texts(args.debug_suspect_texts)
    if args.debug:
        logger.info("Advanced debug mode enabled.")
        logger.info("Suspect default-like texts: {}", sorted(suspect_texts))


    logger.info("Loading LPD model: {}", args.plate_model)
    plate_model = YOLO(args.plate_model, task="detect")

    logger.info("Loading OCR model: {}", args.char_model)
    char_model  = YOLO(args.char_model,  task="detect")

    llm_reader  = None if args.no_vlm else LLMPlateReader()

    # Per-track vote buffers  {track_id: VoteBuffer}
    vote_buffers: dict[int, VoteBuffer] = {}
    track_last_seen: dict[int, int] = {}
    track_best_inferences: dict[int, dict] = {}

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.input}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("{}x{}  {:.1f} fps  {} frames", W, H, fps, total)
    if not args.no_preview:
        logger.info("Live preview ON; press Q or ESC to stop early.")


    tracker_fps = fps / max(args.skip, 1)
    byte_tracker = create_plate_tracker(tracker_fps)
    vote_buffer_ttl_frames = max(1, int(round(fps * TRACK_BUFFER_SECONDS * 2)))

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H),
    )

    WINDOW_NAME = "ALPR – Live Preview  (Q / ESC to quit)"
    if not args.no_preview:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if args.debug:
        cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)


    ocr_frame_cache: dict = {}

    frame_idx  = 0
    last_frame = None
    t_start    = time.time()
    keep_running = True

    while keep_running:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if args.skip > 1 and (frame_idx % args.skip) != 1:
            display = last_frame if last_frame is not None else frame
            writer.write(display)
            if not args.no_preview:
                keep_running = show_preview(WINDOW_NAME, display,
                                            args.preview_scale)
            continue

        annotated = frame.copy()
        ocr_frame_cache.clear()

        det_results = plate_model.predict(
            frame, verbose=False, conf=args.plate_conf
        )

        raw_plate_xyxy  = []   # (x1,y1,x2,y2)
        raw_plate_confs = []
        vehicle_boxes   = []

        for result in det_results:
            for box in result.boxes:
                cls_name = plate_model.names[int(box.cls[0])].lower()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if cls_name in PLATE_CLASSES:
                    raw_plate_xyxy.append([x1, y1, x2, y2])
                    raw_plate_confs.append(conf)
                elif cls_name in VEHICLE_CLASSES and not args.no_vehicles:
                    vehicle_boxes.append((x1, y1, x2, y2, cls_name))

        if raw_plate_xyxy:
            sv_dets = sv.Detections(
                xyxy       = np.array(raw_plate_xyxy,  dtype=np.float32),
                confidence = np.array(raw_plate_confs, dtype=np.float32),
                class_id   = np.zeros(len(raw_plate_xyxy), dtype=int),
            )
        else:
            sv_dets = sv.Detections.empty()

        tracked_dets = byte_tracker.update_with_detections(sv_dets)
        if args.debug:
            logger.debug(
                "frame={} raw_plates={} tracked_plates={} vehicles={}",
                frame_idx, len(raw_plate_xyxy), len(tracked_dets), len(vehicle_boxes),
            )

        for (x1, y1, x2, y2, cls_name) in vehicle_boxes:
            draw_labeled_box(annotated, x1, y1, x2, y2,
                             cls_name.upper(), COLOR_VEHICLE, (0, 0, 0))

        active_track_ids = set()
        for i in range(len(tracked_dets)):
            x1, y1, x2, y2 = map(int, tracked_dets.xyxy[i])
            track_id = coerce_track_id(tracked_dets.tracker_id, i)
            if track_id is None:
                continue

            active_track_ids.add(track_id)
            track_last_seen[track_id] = frame_idx

            # Ensure this track has a vote buffer
            if track_id not in vote_buffers:
                vote_buffers[track_id] = VoteBuffer(
                    maxlen=args.vote_window,
                    min_chars=args.min_chars,
                )

            bbox = (x1, y1, x2, y2)
            plate_conf = get_detection_confidence(tracked_dets, i)

            # Run OCR once per unique track/bbox per frame.
            cache_key = (track_id, x1, y1, x2, y2)
            if cache_key not in ocr_frame_cache:
                should_infer, infer_reason = should_run_track_inference(
                    track_best_inferences, track_id, plate_conf, args,
                    suspect_texts,
                )
                if not should_infer:
                    record = cached_inference_record(
                        track_best_inferences, track_id, frame_idx, bbox,
                        plate_conf, infer_reason,
                    )
                    if args.debug:
                        logger.debug(
                            "frame={} track={} bbox={} skip={} text={} score={:.2f}",
                            frame_idx, track_id, bbox, infer_reason,
                            record.get("text", ""), record.get("score", 0.0),
                        )
                else:
                    pad = 10
                    roi = frame[max(y1 - pad, 0):min(y2 + pad, H),
                                 max(x1 - pad, 0):min(x2 + pad, W)]

                    if roi.size == 0:
                        record = {
                            "text": "", "is_vlm": False, "length": 0,
                            "ocr_length": 0, "min_conf": 0.0,
                            "plate_conf": plate_conf, "frame_idx": frame_idx,
                            "bbox": bbox, "infer_was_run": False,
                            "decision": "empty ROI",
                        }
                        logger.warning("frame={} track={} empty ROI bbox={}",
                                       frame_idx, track_id, bbox)
                    else:
                        if args.debug:
                            roi_routed, roi_prep = preprocess_roi_steps(roi)
                            text, min_conf, length, ocr_debug = run_ocr(
                                roi_prep, char_model, args.conf,
                                return_debug=True,
                            )
                        else:
                            roi_routed = None
                            roi_prep = preprocess_roi(roi)
                            text, min_conf, length = run_ocr(
                                roi_prep, char_model, args.conf,
                            )
                            ocr_debug = {}

                        ocr_text = text
                        is_vlm = False
                        vlm_triggered = False
                        vlm_reason = ""
                        vlm_text = ""

                        if length < args.min_chars and llm_reader is not None:
                            vlm_triggered = True
                            vlm_reason = (f"OCR length {length} < "
                                          f"min chars {args.min_chars}")
                            vlm_text = llm_reader.infer(roi_prep)
                            if vlm_text:
                                text, is_vlm = vlm_text, True

                        record = {
                            "text": text,
                            "ocr_text": ocr_text,
                            "is_vlm": is_vlm,
                            "length": len(text) if text else length,
                            "ocr_length": length,
                            "min_chars": args.min_chars,
                            "min_conf": min_conf,
                            "plate_conf": plate_conf,
                            "frame_idx": frame_idx,
                            "bbox": bbox,
                            "roi": roi,
                            "roi_routed": roi_routed,
                            "roi_processed": roi_prep,
                            "ocr_debug": ocr_debug,
                            "vlm_triggered": vlm_triggered,
                            "vlm_reason": vlm_reason,
                            "vlm_text": vlm_text,
                            "infer_was_run": True,
                            "decision": infer_reason,
                        }

                        if args.debug:
                            record["suspect"] = is_suspect_text(text,
                                                                  suspect_texts)
                            record, _best_changed, decision = update_track_best(
                                track_best_inferences, track_id, record, args,
                                suspect_texts,
                            )
                            record["decision"] = decision
                            if record.get("superseded_by_cached"):
                                record["text"] = record.get("cached_text", "")
                                record["is_vlm"] = record.get("cached_is_vlm", False)
                                record["score"] = record.get("cached_score", 0.0)

                            log_method = logger.warning if record.get("suspect") else logger.debug
                            log_method(
                                "frame={} track={} bbox={} lpd_conf={:.3f} "
                                "ocr='{}' len={} min_conf={:.3f} vlm={} "
                                "vlm_text='{}' effective='{}' score={:.2f} decision={}",
                                frame_idx, track_id, bbox, plate_conf,
                                ocr_text, length, min_conf, vlm_triggered,
                                vlm_text, record.get("text", ""),
                                record.get("score", 0.0), record.get("decision", ""),
                            )

                ocr_frame_cache[cache_key] = record

            record = ocr_frame_cache[cache_key]
            raw_text = record.get("text", "")
            raw_is_vlm = record.get("is_vlm", False)

            # Feed reading into vote buffer
            buf = vote_buffers[track_id]
            should_vote = (record.get("infer_was_run", True) and
                           not record.get("superseded_by_cached", False) and
                           not record.get("suspect", False))
            if should_vote:
                buf.add(raw_text, raw_is_vlm, record.get("min_conf", 0.0))
            elif args.debug:
                logger.debug(
                    "frame={} track={} vote skipped for cached/suspect text='{}'",
                    frame_idx, track_id, raw_text,
                )

            stable_text, stable_is_vlm = buf.best()
            if buf.is_locked():
                display_text = stable_text
                txt_color    = (COLOR_TEXT_VLM if stable_is_vlm
                                else COLOR_TEXT_LOCK)
            else:
                display_text = stable_text or raw_text
                display_is_vlm = (stable_is_vlm if stable_text else raw_is_vlm)
                if display_is_vlm:
                    txt_color = COLOR_TEXT_VLM
                elif len(display_text) >= args.min_chars:
                    txt_color = COLOR_TEXT_OK
                else:
                    txt_color = COLOR_TEXT_WEAK

            draw_labeled_box(annotated, x1, y1, x2, y2,
                             display_text or "?", COLOR_PLATE, txt_color)

            if args.debug and record.get("infer_was_run"):
                keep_running = show_debug_inference(annotated, record,
                                                    track_id, args)
                if not keep_running:
                    break

        if not keep_running:
            break

        prune_stale_vote_buffers(vote_buffers, track_last_seen, frame_idx,
                                 vote_buffer_ttl_frames,
                                 track_best_inferences)

        elapsed  = time.time() - t_start
        live_fps = frame_idx / elapsed if elapsed > 0 else 0
        n_tracks = len(active_track_ids)
        hud      = (f"Frame {frame_idx}/{total}  |  "
                    f"{live_fps:.1f} fps  |  tracks: {n_tracks}")
        cv2.putText(annotated, hud, (10, H - 12),
                    FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(annotated)
        last_frame = annotated

        if not args.no_preview:
            keep_running = show_preview(WINDOW_NAME, annotated,
                                        args.preview_scale)

        if frame_idx % 50 == 0:
            pct = frame_idx / max(total, 1) * 100
            logger.info("{:.1f}%  frame {}/{}  {:.1f} fps",
                        pct, frame_idx, total, live_fps)

    cap.release()
    writer.release()
    if not args.no_preview or args.debug:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    logger.success("Done: {} frames in {:.1f}s ({:.1f} fps avg)",
                   frame_idx, elapsed, avg_fps)
    logger.success("Saved: {}", args.output)



def parse_args():
    p = argparse.ArgumentParser(
        description="ALPR Video Demo – YOLO LPD + OCR + VLM fallback + ByteTrack (supervision)"
    )
    p.add_argument("--input",         default=str(DEFAULT_VIDEO_INPUT),
                   help="Input video file (default: data/videos/video.mp4)")
    p.add_argument("--output",        default=str(DEFAULT_VIDEO_OUTPUT),
                   help="Output video file (default: data/inference/videos/video.mp4)")
    p.add_argument("--plate-model",   default=str(DEFAULT_PLATE_MODEL),
                   help="LPD YOLO weights")
    p.add_argument("--char-model",    default=str(DEFAULT_CHAR_MODEL),
                   help="Char OCR YOLO weights")
    p.add_argument("--conf",          type=float, default=0.15,
                   help="OCR confidence threshold (default: 0.15)")
    p.add_argument("--plate-conf",    type=float, default=0.50,
                   help="Plate detection confidence (default: 0.50)")
    p.add_argument("--min-chars",     type=int,   default=6,
                   help="Chars below this trigger VLM fallback (default: 6)")
    p.add_argument("--skip",          type=int,   default=1,
                   help="Process every N-th frame; duplicate others (default: 1)")
    p.add_argument("--no-vlm",        action="store_true",
                   help="Disable VLM fallback")
    p.add_argument("--no-vehicles",   action="store_true",
                   help="Skip vehicle bounding boxes")
    # Preview flags
    p.add_argument("--no-preview",    action="store_true",
                   help="Disable live preview window (headless mode)")
    p.add_argument("--preview-scale", type=float, default=1.0,
                   help="Scale factor for preview window (default: 1.0). "
                        "Use 0.5 for half-size preview on small screens.")
    # vote buffer
    p.add_argument("--vote-window",   type=int,   default=20,
                   help="Max OCR readings kept per track for voting (default: 20)")
    # advanced debug mode
    p.add_argument("--debug", action="store_true",
                   help="Enable advanced per-inference debug UI and logs")
    p.add_argument("--debug-no-pause", action="store_true",
                   help="Do not pause after each debug inference")
    p.add_argument("--debug-window-scale", type=float, default=1.0,
                   help="Scale the debug grid window (default: 1.0)")
    p.add_argument("--debug-save-dir", default="",
                   help="Optional directory to save debug grid images")
    p.add_argument("--debug-no-track-cache", action="store_true",
                   help="Disable debug track-level inference reuse")
    p.add_argument("--debug-min-conf-delta", type=float, default=0.05,
                   help="LPD confidence improvement needed to reinfer a cached track")
    p.add_argument("--debug-min-score-delta", type=float, default=0.10,
                   help="Score improvement needed to replace a cached inference")
    p.add_argument("--debug-suspect-texts", default="ABC1234",
                   help="Comma-separated placeholder-like OCR/VLM outputs to warn about")
    return p.parse_args()


if __name__ == "__main__":
    process_video(parse_args())
