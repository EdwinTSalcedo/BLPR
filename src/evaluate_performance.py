"""Evaluate BLPR predictions against labels.csv and print ablation metrics."""

import argparse
import difflib
from pathlib import Path

import cv2
import pandas as pd
from colorama import Fore, Style, init

from blpr_pipeline import (
    CHAR_MODEL,
    CONF_DROP_RATIO,
    LLMPlateReader,
    MIN_CHARS_THRESHOLD,
    PLATE_MODEL,
    PICTURES_DIR,
    context_heur_ad_gamma,
    run_ocr,
    smart_perspective_router,
)

LABELS_FILE = "labels.csv"
RUN_DUAL_OCR = False
CHAR_MODEL_1 = CHAR_MODEL
CHAR_MODEL_2 = CHAR_MODEL
VALID_DISTANCES = {"CERCA", "MEDIO"}
VALID_ANGLES = {"NORMAL", "INCLINADO", "PRONUNCIADO"}
CONFIGS = ["Raw", "No_Illum", "No_Skew", "No_LLM", "Full"]
MODELS = ["OCR_2", "OCR_1"] if RUN_DUAL_OCR else ["OCR_2"]


def create_stats_dict():
    return {'TP_Chars': 0, 'GT_Chars': 0, 'Det_Chars': 0, 'Total_Sim': 0.0, 'Total_Plates': 0}

def get_matching_chars_count(s1, s2):
    matcher = difflib.SequenceMatcher(None, s1, s2)
    return sum(n for i, j, n in matcher.get_matching_blocks())

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the BLPR evaluation pipeline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PICTURES_DIR,
        help="Folder containing labels.csv and the evaluation images (default: data/pictures/).",
    )
    args = parser.parse_args(argv)
    data_dir = args.data_dir.expanduser()

    labels_path = data_dir / LABELS_FILE
    if not labels_path.exists():
        parser.error(f"Labels file not found at {labels_path}")
    df = pd.read_csv(labels_path)

    from ultralytics import YOLO

    init(autoreset=True)
    ocr_model_path = CHAR_MODEL_2
    if not ocr_model_path.exists():
        print(f"Optional OCR model not found at {ocr_model_path}. Falling back to {CHAR_MODEL_1}.")
        ocr_model_path = CHAR_MODEL_1
    print(f"{Fore.YELLOW}Loading YOLO Models...{Style.RESET_ALL}")
    plate_model = YOLO(str(PLATE_MODEL), task="detect")
    char_model_2 = YOLO(str(ocr_model_path), task="detect")
    if RUN_DUAL_OCR:
        char_model_1 = YOLO(str(CHAR_MODEL_1), task="detect")
    llm_reader = LLMPlateReader()

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
        img_path = data_dir / img_filename
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
