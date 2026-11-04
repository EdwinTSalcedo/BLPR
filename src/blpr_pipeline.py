"""Shared OCR, preprocessing, and VLM fallback for BLPR image workflows."""

from pathlib import Path
import warnings

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PICTURES_DIR = REPO_ROOT / "data" / "pictures"
INFERENCE_DIR = REPO_ROOT / "data" / "inference"
INFERENCE_PICTURES_DIR = INFERENCE_DIR / "pictures"
MODELS_DIR = REPO_ROOT / "models"
PLATE_MODEL = MODELS_DIR / "best_lpd.pt"
CHAR_MODEL = MODELS_DIR / "best_lpr.pt"
CONFIDENCE_THRESHOLD = 0.15
IGNORED_STRINGS = ["BOLIVIA", "_"]
MIN_CHARS_THRESHOLD = 6
CONF_DROP_RATIO = 0.3
LLM_GEMA_ID = "gemma3:4b"


class LLMPlateReader:
    def __init__(self):
        from agno.agent import Agent
        from agno.models.ollama import Ollama

        print("Initializing Gemma 3 fallback...")
        self.agent = Agent(
            model=Ollama(id=LLM_GEMA_ID),
            description="You are a precise License Plate Recognition system.",
            instructions=[
                "Return ONLY the alphanumeric characters seen on the plate.",
                "Do not include country names like BOLIVIA.",
                "No markdown, no explanations, no spaces.",
            ],
        )
        self._warned = False

    def get_image_bytes(self, image):
        if image is None or image.size == 0:
            return None
        success, buffer = cv2.imencode(".jpg", image)
        return buffer.tobytes() if success else None

    def infer(self, image):
        from agno.media import Image as AgnoImage

        try:
            img_bytes = self.get_image_bytes(image)
            if not img_bytes:
                return ""
            response = self.agent.run(
                "Read this license plate text.",
                images=[AgnoImage(content=img_bytes)],
            )
            text = (response.content or "").strip().upper()
            return "".join(c for c in text if c.isalnum())
        except Exception as error:
            if not self._warned:
                warnings.warn(
                    f"Gemma fallback failed; keeping the OCR result. {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned = True
            return ""


def context_heur_ad_gamma(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_v = np.mean(v)
    std_v = np.std(v)

    if std_v > 60: return image

    if 80 <= mean_v <= 160: return image

    gamma = np.log(128.0 / 255.0) / np.log(mean_v / 255.0)

    gamma = np.clip(gamma, 0.6, 1.5)

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    v_gamma = cv2.LUT(v, table)

    return cv2.cvtColor(cv2.merge([h, s, v_gamma]), cv2.COLOR_HSV2BGR)


def illumination_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def apply_nlm_denoise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def apply_heur_3d_morph(roi):
    h, w = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    block_size = max(3, int(h * 0.35) | 1)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 10)

    mask = np.zeros_like(binary)
    mask[int(h * 0.20):int(h * 0.90), int(w * 0.05):int(w * 0.95)] = 255
    binary_masked = cv2.bitwise_and(binary, mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    morphed = cv2.morphologyEx(binary_masked, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return roi

    largest_contour = max(contours, key=cv2.contourArea)
    blob_area = cv2.contourArea(largest_contour)
    if not (0.10 * (w * h) <= blob_area <= 0.60 * (w * h)): return roi

    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(blob_area) / hull_area if hull_area > 0 else 0
    if solidity < 0.45: return roi

    rect = cv2.minAreaRect(largest_contour)
    box = np.int32(cv2.boxPoints(rect))
    ordered_pts = order_points(box)

    bx, by, bw, bh = cv2.boundingRect(ordered_pts)
    dst_pts = np.array([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
    return cv2.warpPerspective(roi, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def smart_perspective_router(roi):
    h, w = roi.shape[:2]

    scout_img = illumination_clahe(roi)
    gray = cv2.cvtColor(scout_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return roi

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < (w * h * 0.15): return roi

    epsilon = 0.05 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        ordered_pts = order_points(np.float32(approx.reshape(4, 2)))

        widthA = np.linalg.norm(ordered_pts[2] - ordered_pts[3])
        widthB = np.linalg.norm(ordered_pts[1] - ordered_pts[0])
        heightA = np.linalg.norm(ordered_pts[1] - ordered_pts[2])
        heightB = np.linalg.norm(ordered_pts[0] - ordered_pts[3])

        minWidth, maxWidth = min(widthA, widthB), max(widthA, widthB)
        minHeight, maxHeight = min(heightA, heightB), max(heightA, heightB)

        if minWidth == 0 or minHeight == 0: return roi

        width_ratio = maxWidth / minWidth
        height_ratio = maxHeight / minHeight
        max_foreshortening = max(width_ratio, height_ratio)

        dx = ordered_pts[1][0] - ordered_pts[0][0]
        dy = ordered_pts[1][1] - ordered_pts[0][1]
        angle = abs(np.degrees(np.arctan2(dy, dx)))

        if max_foreshortening > 1.15 or angle > 15.0:
            bx, by, bw, bh = cv2.boundingRect(ordered_pts)
            dst_pts = np.array([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]], dtype="float32")

            max_disp = np.max(np.linalg.norm(ordered_pts - dst_pts, axis=1))
            if max_disp > (w * 0.25): return roi

            M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
            return cv2.warpPerspective(roi, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        elif max_foreshortening < 1.06 and angle < 5.0:
            denoised_roi = apply_nlm_denoise(roi)
            return apply_heur_3d_morph(denoised_roi)

    return roi

def run_ocr(roi_img, model):
    roi_resized = cv2.resize(roi_img, (1020, 730), interpolation=cv2.INTER_CUBIC)

    results = model.predict(roi_resized, verbose=False, conf=CONFIDENCE_THRESHOLD)
    chars = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            char_str = model.names[cls_id]

            if char_str.upper() in IGNORED_STRINGS or "_" in char_str:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            chars.append({
                'char': char_str,
                'cx': (x1 + x2) / 2,
                'cy': (y1 + y2) / 2,
                'h': y2 - y1,
                'conf': box.conf[0].item()
            })

    if not chars: return "", 0.0, 0.0, 0

    chars.sort(key=lambda x: x['cy'])
    lines = []
    current_line = [chars[0]]

    for char in chars[1:]:
        prev_char = current_line[-1]
        if (char['cy'] - prev_char['cy']) > (prev_char['h'] * 0.5):
            lines.append(current_line)
            current_line = [char]
        else:
            current_line.append(char)
    lines.append(current_line)

    final_text = ""
    confs = []
    for line in lines:
        line.sort(key=lambda x: x['cx'])
        for char in line:
            final_text += char['char']
            confs.append(char['conf'])

    return final_text, max(confs), min(confs), len(final_text)


def recognize_plate(roi, char_model, llm_reader=None):
    """Apply the full preprocessing/OCR pipeline to a detected plate crop."""
    prepared = context_heur_ad_gamma(smart_perspective_router(roi.copy()))
    text, max_conf, min_conf, length = run_ocr(prepared, char_model)
    if llm_reader is not None and (
        length < MIN_CHARS_THRESHOLD
        or (length > 0 and min_conf < max_conf * CONF_DROP_RATIO)
    ):
        text = llm_reader.infer(prepared) or text
    return text
