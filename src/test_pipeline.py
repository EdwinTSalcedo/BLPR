import numpy as np
import cv2
from ultralytics import YOLO
import pandas as pd
import os
import difflib
from pathlib import Path
from colorama import Fore, Style, init

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.media import Image as AgnoImage

<<<<<<< HEAD
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_FOLDER = REPO_ROOT / "dataset"
LABELS_FILE = "labels.csv"
CONFIDENCE_THRESHOLD = 0.2
IGNORED_STRINGS = ["BOLIVIA", "_"]

MODELS_DIR = REPO_ROOT / "models"
CHAR_MODEL_1 = MODELS_DIR / "best_lpr.pt"
CHAR_MODEL_2 = MODELS_DIR / "best_lpr_4.pt"
PLATE_MODEL = MODELS_DIR / "best_lpd.pt"
=======

DATASET_FOLDER = 'dataset'
LABELS_FILE = 'labels.csv' 
CONFIDENCE_THRESHOLD = 0.15
IGNORED_STRINGS = ["BOLIVIA", "_"]

RUN_DUAL_OCR = False

CHAR_MODEL_2 = "best_lpr_5.pt"
CHAR_MODEL_1 = "best_lpr_5.pt"
PLATE_MODEL = "best_lpd.pt"
>>>>>>> d94e4e63de6910d0c9c14eaa9e6656ae2d2d9d97

MIN_CHARS_THRESHOLD = 6        
CONF_DROP_RATIO = 0.3         
LLM_GEMA_ID = "gemma3:4b"

VALID_DISTANCES = {"CERCA", "MEDIO"}
VALID_ANGLES = {"NORMAL", "INCLINADO", "PRONUNCIADO"}
CONFIGS = ["Raw", "No_Illum", "No_Skew", "No_LLM", "Full"]

MODELS = ["OCR_2", "OCR_1"] if RUN_DUAL_OCR else ["OCR_2"]

LLM_SYS_DESCRIPTION = "You are a precise License Plate Recognition system."
LLM_SYS_INSTRUCTIONS = [
    "Return ONLY the alphanumeric characters seen on the plate.", 
    "Do not include country names like BOLIVIA.", 
    "No markdown, no explanations, no spaces."
]

init(autoreset=True)

if not CHAR_MODEL_2.exists():
    print(f"{Fore.YELLOW}Optional OCR model not found at {CHAR_MODEL_2}. Falling back to {CHAR_MODEL_1}.{Style.RESET_ALL}")
    CHAR_MODEL_2 = CHAR_MODEL_1

print(f"{Fore.YELLOW}Loading YOLO Models...{Style.RESET_ALL}")
<<<<<<< HEAD
plate_model = YOLO(str(PLATE_MODEL), task='detect')
char_model_1 = YOLO(str(CHAR_MODEL_1), task='detect')
char_model_2 = YOLO(str(CHAR_MODEL_2), task='detect')
=======
plate_model = YOLO(PLATE_MODEL, task='detect') 
char_model_2 = YOLO(CHAR_MODEL_2, task='detect')
if RUN_DUAL_OCR:
    char_model_1 = YOLO(CHAR_MODEL_1, task='detect')
>>>>>>> d94e4e63de6910d0c9c14eaa9e6656ae2d2d9d97

class LLMPlateReader:
    def __init__(self):
        print(f"{Fore.YELLOW}Initializing Gemma 3 Fallback Agent...{Style.RESET_ALL}")
        self.agent = Agent(
            model=Ollama(id=LLM_GEMA_ID),
            description=LLM_SYS_DESCRIPTION,
            instructions=LLM_SYS_INSTRUCTIONS
        )
            
    def get_image_bytes(self, image):
        if image is None or image.size == 0: return None
        success, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes() if success else None

    def infer(self, image):
        if not self.agent: return ""
        try:
            img_bytes = self.get_image_bytes(image)
            if not img_bytes: return ""
            response = self.agent.run("Read this license plate text.", images=[AgnoImage(content=img_bytes)])
            text = response.content.strip().upper()
            return ''.join(c for c in text if c.isalnum())
        except Exception:
            return ""

llm_reader = LLMPlateReader()


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

def create_stats_dict():
    return {'TP_Chars': 0, 'GT_Chars': 0, 'Det_Chars': 0, 'Total_Sim': 0.0, 'Total_Plates': 0}

def get_matching_chars_count(s1, s2):
    matcher = difflib.SequenceMatcher(None, s1, s2)
    return sum(n for i, j, n in matcher.get_matching_blocks())

def main():
    labels_path = DATASET_FOLDER / LABELS_FILE
    if not labels_path.exists():
        print(f"Error: Labels file not found at {labels_path}")
        return
    df = pd.read_csv(labels_path)

    stats = {
        model_name: {
            config: {
                "Overall": create_stats_dict(),
                "CERCA": create_stats_dict(), "MEDIO": create_stats_dict(),
                "NORMAL": create_stats_dict(), "INCLINADO": create_stats_dict(), "PRONUNCIADO": create_stats_dict()
            } for config in CONFIGS
        } for model_name in MODELS
    }

    print(f"\n{Fore.CYAN}{'='*130}")
    ocr_mode_str = "DUAL OCR MODELS" if RUN_DUAL_OCR else "SINGLE OCR MODEL"
    print(f"STARTING ULTIMATE PIPELINE VALIDATION BENCHMARK ({ocr_mode_str})")
    print(f"{'='*130}{Style.RESET_ALL}\n")

    header = f"{'Img #':<5} | {'GT':<10} | {'Model':<7} | {'Raw':<9} | {'No Illum':<9} | {'No Skew':<9} | {'No LLM':<9} | {'Full':<9} | {'LLM Trig?':<9}"
    print(header)
    print("-" * 130)

    img_count = 0
    lpd_misses = 0
    llm_triggers = {m: 0 for m in MODELS}
    total_valid_plates = 0

    model_refs = {"OCR_2": char_model_2}
    if RUN_DUAL_OCR:
        model_refs["OCR_1"] = char_model_1

    for index, row in df.iterrows():
        img_filename = row['imagen']
        img_path = DATASET_FOLDER / img_filename
        if not img_path.exists(): continue

        ground_truths = []
        angle_val = str(row.get('angle', 'UNKNOWN')).strip().upper()
        for i in range(1, 6):
            p_col, d_col = f'plate{i}', f'distance{i}'
            if p_col in row and pd.notna(row[p_col]) and str(row[p_col]).strip():
                dist_val = str(row.get(d_col, 'UNKNOWN')).strip().upper()
                if dist_val == "LEJOS": continue
                ground_truths.append({"text": str(row[p_col]).strip(), "dist": dist_val, "angle": angle_val})

        if not ground_truths: continue 
        img_count += 1
        
        original_img = cv2.imread(str(img_path))
        plate_results = plate_model.predict(original_img, verbose=False, conf=0.5)
        
        rois = []
        for result in plate_results:
            for box in result.boxes:
                if plate_model.names[int(box.cls)].lower() in ["plate", "placa"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    pad_x1, pad_y1 = max(x1-10, 0), max(y1-10, 0)
                    roi = original_img[pad_y1:y2+10, pad_x1:x2+10]
                    rois.append(roi)

        if not rois:
            lpd_misses += len(ground_truths)
            for gt in ground_truths:
                print(f"{img_count:<5} | {Fore.RED}{gt['text']:<10}{Style.RESET_ALL} | {'N/A':<7} | {'-':<9} | {'-':<9} | {'-':<9} | {'-':<9} | {'-':<9} | LPD Failed")
            continue

        pipeline_results = []
        for roi in rois:
            
            roi_ill_only = context_heur_ad_gamma(roi.copy())
            roi_routed_raw = smart_perspective_router(roi.copy())
            
            roi_routed_then_ill = context_heur_ad_gamma(roi_routed_raw.copy())

            roi_metrics = {}
            for m_name in MODELS:
                model = model_refs[m_name]
                
                text_raw, _, _, _ = run_ocr(roi, model)
                text_no_ill, _, _, _ = run_ocr(roi_routed_raw, model)
                text_no_skew, _, _, _ = run_ocr(roi_ill_only, model)
                text_no_llm, max_conf, min_conf, length = run_ocr(roi_routed_then_ill, model)

                final_text = text_no_llm
                llm_triggered = False
                
                if length < MIN_CHARS_THRESHOLD or (length > 0 and min_conf < (max_conf * CONF_DROP_RATIO)):
                    llm_triggered = True
                    llm_text = llm_reader.infer(roi_routed_then_ill)
                    if llm_text: final_text = llm_text
                
                roi_metrics[m_name] = {
                    "Raw": text_raw,
                    "No_Illum": text_no_ill,
                    "No_Skew": text_no_skew,
                    "No_LLM": text_no_llm,
                    "Full": final_text,
                    "Triggered": llm_triggered
                }
                
            pipeline_results.append(roi_metrics)

        for gt in ground_truths:
            gt_text, dist, angle = gt["text"], gt["dist"], gt["angle"]
            total_valid_plates += 1
            
            best_sim, best_idx = -1.0, -1
            
            for d_idx, res in enumerate(pipeline_results):
                max_sim = max([difflib.SequenceMatcher(None, gt_text, res[m]["Full"]).ratio() for m in MODELS])
                if max_sim > best_sim:
                    best_sim, best_idx = max_sim, d_idx
                    
            if best_idx != -1: 
                matched_res = pipeline_results.pop(best_idx)
            else:
                matched_res = {m: {c: "" for c in CONFIGS} | {"Triggered": False} for m in MODELS}

            categories = ["Overall"]
            if dist in VALID_DISTANCES: categories.append(dist)
            if angle in VALID_ANGLES: categories.append(angle)

            for m_idx, m_name in enumerate(MODELS):
                m_data = matched_res[m_name]
                
                if m_data.get("Triggered"):
                    llm_triggers[m_name] += 1

                for config in CONFIGS:
                    det_text = m_data[config]
                    matches = get_matching_chars_count(gt_text, det_text)
                    sim = difflib.SequenceMatcher(None, gt_text, det_text).ratio()
                    
                    for cat in categories:
                        s = stats[m_name][config][cat]
                        s['GT_Chars'] += len(gt_text); s['Det_Chars'] += len(det_text)
                        s['TP_Chars'] += matches; s['Total_Sim'] += sim; s['Total_Plates'] += 1

                img_str = str(img_count) if m_idx == 0 else ""
                gt_str = gt_text if m_idx == 0 else ""
                
                llm_str = f"{Fore.MAGENTA}Yes{Style.RESET_ALL}" if m_data["Triggered"] else "No"
                f_col = Fore.GREEN if difflib.SequenceMatcher(None, gt_text, m_data["Full"]).ratio() > 0.8 else Fore.RED
                
                print(f"{img_str:<5} | {gt_str:<10} | {Fore.YELLOW}{m_name:<7}{Style.RESET_ALL} | {m_data['Raw']:<9} | {m_data['No_Illum']:<9} | {m_data['No_Skew']:<9} | {m_data['No_LLM']:<9} | {f_col}{m_data['Full']:<9}{Style.RESET_ALL} | {llm_str:<9}")
            print("-" * 130)

    def print_ablation_table(model_name, cat_key, title):
        print(f"\n{title} METRICS")
        print(f"{'Configuration':<15} | {'Plates':<6} | {'Avg Sim':<7} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
        print("-" * 75)
        
        for config in CONFIGS:
            s = stats[model_name][config][cat_key]
            if s['Total_Plates'] == 0: continue
            
            avg_sim = s['Total_Sim'] / s['Total_Plates']
            prec = s['TP_Chars'] / s['Det_Chars'] if s['Det_Chars'] > 0 else 0
            rec = s['TP_Chars'] / s['GT_Chars'] if s['GT_Chars'] > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            color = Fore.CYAN if config == "Full" else ""
            reset = Style.RESET_ALL if config == "Full" else ""
            
            print(f"{color}{config:<15}{reset} | {s['Total_Plates']:<6} | {avg_sim:.4f}  | {prec:.4f}     | {rec:.4f}     | {f1:.4f}")

    total_gts = stats[MODELS[0]]["Full"]["Overall"]["Total_Plates"] + lpd_misses

    for m_name in MODELS:
        print("\n\n" + "#"*85)
        print(f"FINAL ABLATION BENCHMARK RESULTS: {Fore.YELLOW}{m_name}{Style.RESET_ALL}")
        print("#"*85)
        
        if total_gts > 0:
            print(f"\nLPD Misses: {lpd_misses} / {total_gts} GT Plates (Excluding LEJOS)")
            
        if total_valid_plates > 0:
            llm_percentage = (llm_triggers[m_name] / total_valid_plates) * 100
            print(f"LLM Fallback Usage: {llm_triggers[m_name]}/{total_valid_plates} ({llm_percentage:.1f}%)\n")

        print_ablation_table(m_name, "Overall", "OVERALL PIPELINE")
        print("=" * 75)
        for cat in VALID_DISTANCES: print_ablation_table(m_name, cat, f"DISTANCE: {cat}")
        print("=" * 75)
        for cat in VALID_ANGLES: print_ablation_table(m_name, cat, f"ANGLE: {cat}")

if __name__ == "__main__":
    main()
