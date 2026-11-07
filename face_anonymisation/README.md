# Face anonymisation

This folder contains my scripts for anonymising faces in BLPR-A and preparing the public BLPR-D release. The face-anonymisation script runs locally and preserves the input split folders and label files.

## Setup

Activate the `blpr` Conda environment. From the repository root, install the dependencies:

```bash
conda activate blpr
python -m pip install -r face_anonymisation/requirements.txt
```

## Model files

Download both detector weights and put them in `face_anonymisation/models/`. Git ignores the model files, and the scripts do not download them automatically.

- **YuNet:** `face_detection_yunet_2023mar.onnx` from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
- **SCRFD:** `det_2.5g.onnx`, extracted from `buffalo_m.zip` in the [InsightFace model zoo](https://github.com/deepinsight/insightface/releases/tag/model-zoo).

SCRFD runs with ONNX Runtime's CPU provider. These face models are used only for detection; no face-recognition or identity models are loaded. Check the InsightFace model terms before redistributing its weights or using them commercially.

## Run on BLPR-A

Run this command from the repository root:

```bash
python face_anonymisation/anonymise_faces.py \
  --input-dir ../BLPR-datasets/BLPR-A \
  --output-dir ../BLPR-datasets/BLPR-A-anonymised \
  --detector both \
  --yunet-model face_anonymisation/models/face_detection_yunet_2023mar.onnx \
  --scrfd-model face_anonymisation/models/det_2.5g.onnx \
  --jpeg-quality 95 \
  --protect-yolo-class-id 1 \
  --protected-overlap-threshold 0.25 \
  --padding 0.30 \
  --blur-strength 0.5
```

In BLPR-A, YOLO class 1 is the plate class. A face detection is discarded if it overlaps at least 25% of an annotated plate box. After blurring, the script restores the original pixels inside protected plate boxes.

Use `--dry-run --limit 1` to check detections without saving processed images. Use `--review-dir PATH` to save images with the detections marked for review.

## Output

For the numbered public BLPR-D dataset, use [Prepare BLPR-D for release](BLPR-D.md).

The script keeps the input split folders, copies label files and `data.yaml`, and writes `face_anonymisation_report.csv` and `face_anonymisation_summary.json` in the output folder. The reports include detector results, detections excluded to protect plate annotations, processing status, and errors.

Images with detected faces are saved in their original format; for JPEGs, `--jpeg-quality` sets the output quality (1–100; default 100). Images without retained detections are copied unchanged. Image dimensions are preserved, along with EXIF, ICC profile and DPI metadata when available. Review the output for missed faces and false detections before releasing it.
