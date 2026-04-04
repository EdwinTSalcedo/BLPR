<h1 align="center">BLPR: Robust License Plate Recognition Under Viewpoint and Illumination Variations</h1>

<p align="center">
  <a href="">
    <img src="https://img.shields.io/badge/Paper-PDF-red" alt="Paper"></a>
  <a href="">
    <img src="https://img.shields.io/badge/arXiv-2508.04663-b31b1b" alt="arXiv"></a>
  <a href="">
    <img src="https://img.shields.io/badge/Dataset-Page-blue" alt="Dataset Page"></a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License"></a>
</p>

<p align="center">
  A robust licence plate detection and recognition framework for Bolivian vehicles under real-world conditions.
</p>

**Authors:** [Guillermo Auza](https://www.linkedin.com/in/guillermo-a-358247253/), [Sergio Castro](https://www.linkedin.com/in/sergio-castro-sandoval-663a78347/), [Diego Calvimontes](https://www.linkedin.com/in/calvimontes-diego/), [Natalia Condori](https://www.linkedin.com/in/natalia-condori-864520287/),  and [Edwin Salcedo](https://www.linkedin.com/in/edwinsalcedo)

## Introduction

Vehicle identity verification remains a major challenge in Latin America, contributing to rising cases of stolen vehicles, duplicated plates, and illegal transit. In countries such as Bolivia, the problem is exacerbated by the lack of technological infrastructure and publicly available datasets tailored to local conditions.

Automatic Licence Plate Detection and Recognition (LPDR) systems provide a scalable solution, but their performance is still limited under real-world conditions, particularly due to viewpoint and illumination variations. Additionally, most existing datasets do not explicitly account for these factors, and there is a notable absence of data from underrepresented regions such as the Andean context.

## This Repository

This repository contains the official implementation of the Bolivian license plate recognition (<b>BLPR</b>) framework, designed to operate robustly under perspective distortions and extreme illumination variations. The system uses a two-stage pipeline based on YOLOv11 and YOLOv8x, with an optional fallback mechanism powered by a local Ollama model. The main components are:

- **BLPR Dataset** with explicit categorisation by viewpoint and illumination  
- **Detection & Rectification Pipeline** based on YOLOv11n for accurate localisation and preprocessing  
- **Hybrid OCR Module** combining YOLOv8x with a local Gemma-based fallback for improved recognition under ambiguity  

## Framework Overview
The framework processes images as follows:

1. **Detection:** YOLOv11n locates the vehicle and the license plate.

2. **Preprocessing:** The system validates that the plate is within the vehicle and extracts the ROI with a 10-pixel margin.

3. **Illumination Normalization:** The ROI is converted to YUV space to evaluate average luminance and apply enhancements based on the threshold.

4. **Rectification:** The image is binarized and edges are detected to apply a perspective transformation.

5. **Primary OCR:** YOLOv8x performs character recognition.

6. **Validation and Fallback:** If OCR fails or is unreliable, a local Ollama-hosted `gemma3:4b` model is queried.

<p align="center"><img src="./images/pipeline.jpg" width="90%"></p>


## Getting Started

- [Installation](#installation)
- [Inference](#inference)
- [Dataset](#blpr-dataset)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Citation](#citation)

## Installation

```bash
git clone --recursive git@github.com:EdwinTSalcedo/BLPR.git blpr
cd blpr
```

**Option 1: Using requirements.txt**

```bash
conda create -n blpr python=3.13
conda activate blpr
pip install -r requirements.txt
```

**Option 2: Using pyproject.toml**

```bash
conda create -n blpr python=3.13
conda activate blpr
pip install -e .
```

## Inference

The evaluation pipeline is provided in `src/test_pipeline.py`. It runs the full BLPR workflow: plate detection, illumination enhancement, geometric rectification, OCR, and LLM fallback.

### Expected layout

Before running inference, make sure the repository contains:

```text
BLPR/
├── dataset/
│   ├── labels.csv
│   └── <evaluation_images>
├── models/
│   ├── best_lpd.pt
│   ├── best_lpr.pt
│   └── best_lpr_4.pt
└── src/
    └── test_pipeline.py
```

- `dataset/labels.csv` should contain the image names and ground-truth plate annotations used by the script.
- `models/best_lpd.pt` is the plate detection model.
- `models/best_lpr.pt` and `models/best_lpr_4.pt` are the OCR models compared during evaluation.
- If `models/best_lpr_4.pt` is missing, the script automatically falls back to `models/best_lpr.pt` for both OCR passes.

### Run the pipeline

```bash
python src/test_pipeline.py
```

### What the script does

- Loads the detection and OCR YOLO models.
- Reads evaluation metadata from `dataset/labels.csv`.
- Detects license plates in each image and extracts plate regions.
- Applies illumination correction and geometric rectification.
- Runs OCR with two character-recognition models.
- Triggers the local LLM fallback when OCR confidence is low or incomplete.
- Prints per-image predictions and aggregate ablation metrics by distance and viewing angle.

### LLM fallback

The current implementation uses a local Ollama-compatible model through `agno`, configured as:

```python
LLM_GEMA_ID = "gemma3:4b"
```

Before running the pipeline, make sure the model is available locally in Ollama:

```bash
ollama run gemma3:4b
```

This will pull the model if it is not already installed. The fallback stage depends on Ollama being available locally with `gemma3:4b`; otherwise, you should disable or modify the LLM portion in `src/test_pipeline.py`.

## BLPR Dataset
The BLPR dataset consists of four sub-datasets:
| Dataset | Samples  | Type      | Description                                                                          |
|---------|----------|-----------|--------------------------------------------------------------------------------------|
| BLPR-A  | 3,550    | Real      | Street photos from La Paz with vehicle and license plate annotations.                |
| BLPR-B  | 49,392   | Synthetic | Created in Blender with weather variations and angles from 30 degrees to 150 degrees.|
| BLPR-C  | 1,374    | Real/Aug  | Individual plates for OCR training with class balancing.                             |
| BLPR-D  | 499      | Real      | Validation dataset with luminance (lux) data and real perspective conditions.        |

## Evaluation

The system achieves 88.4% character-level accuracy on real street images under uncontrolled conditions.

| Model    | Task                                      | IoU    | F1-Score | Inference Time       |
|----------|-------------------------------------------|--------|----------|----------------------|
| YOLOv11n | Detection (LPD)                           | 0.7639 | 0.8682   | 3.62 ms              |
| YOLOv8x  | Optical character recognition (LPR)       | 0.9458 | 0.9837   | 29.74 ms             |

## Project Structure

```text
BLPR/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── images/
│   └── pipeline.jpg
├── models/
│   ├── best_lpd.pt
│   └── best_lpr.pt
└── src/
    └── test_pipeline.py
```

- `README.md`: project overview, setup instructions, dataset summary, evaluation results, and citation.
- `LICENSE`: repository license information.
- `pyproject.toml`: Python project metadata and dependency definitions.
- `requirements.txt`: alternative dependency installation file.
- `uv.lock`: locked dependency versions for reproducible environments.
- `images/pipeline.jpg`: architecture figure used in the README.
- `models/best_lpd.pt`: trained YOLO model for license plate detection.
- `models/best_lpr.pt`: trained YOLO model for character recognition.
- `models/best_lpr_4.pt` (optional): second OCR model used for ablation if available.
- `src/test_pipeline.py`: end-to-end inference and evaluation script, including preprocessing, OCR, and LLM fallback logic.

At runtime, the pipeline also expects a `dataset/` directory with evaluation images and a `labels.csv` file, although that data is not currently committed to this repository.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{auza_banegas2026,
  title   = {Robust License Plate Recognition Under Viewpoint and Illumination Variations: A Bolivian Case Study},
  author  = {Auza Banegas, Guillermo J. and Castro Sandoval, Sergio S. and Calvimontes Vera, Diego A. and Condori Peredo, Natalia L. and Salcedo, Edwin},
  journal = {IEEE Latin America Transactions}
}
```
