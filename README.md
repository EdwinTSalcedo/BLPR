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

**Authors:** [Guillermo Auza](https://www.linkedin.com/in/guillermo-j-auza-banegas/), [Sergio Castro](https://www.linkedin.com/in/sergio-s-castro-sandoval/), [Diego Calvimontes](https://www.linkedin.com/in/diego-a-calvimontes-vera/), [Natalia Condori](https://www.linkedin.com/in/natalia-l-condori-peredo/),  and [Edwin Salcedo](https://www.linkedin.com/in/edwinsalcedo)

## Introduction

Vehicle identity verification remains a major challenge in Latin America, contributing to rising cases of stolen vehicles, duplicated plates, and illegal transit. In countries such as Bolivia, the problem is exacerbated by the lack of technological infrastructure and publicly available datasets tailored to local conditions.

Automatic Licence Plate Detection and Recognition (LPDR) systems provide a scalable solution, but their performance is still limited under real-world conditions, particularly due to viewpoint and illumination variations. Additionally, most existing datasets do not explicitly account for these factors, and there is a notable absence of data from underrepresented regions such as the Andean context.

## This Repository

This repository contains the official implementation of the Bolivian license plate recognition (BLPR) framework, designed to operate robustly under perspective distortions and extreme illumination variations. The system uses a two-stage pipeline based on YOLOv11 and YOLOv8x, with an innovative fallback mechanism powered by Llama 3.2. The main components are:

- **BLPR Dataset** with explicit categorisation by viewpoint and illumination  
- **Detection & Rectification Pipeline** based on YOLOv11n for accurate localisation and preprocessing  
- **Hybrid OCR Module** combining YOLOv8x with Llama 3.2 for improved recognition under ambiguity  

## Framework Overview
The framework processes images as follows:

1. **Detection:** YOLOv11n locates the vehicle and the license plate.

2. **Preprocessing:** The system validates that the plate is within the vehicle and extracts the ROI with a 10-pixel margin.

3. **Illumination Normalization:** The ROI is converted to YUV space to evaluate average luminance and apply enhancements based on the threshold.

4. **Rectification:** The image is binarized and edges are detected to apply a perspective transformation.

5. **Primary OCR:** YOLOv8x performs character recognition.

6. **Validation and Fallback:** If OCR fails or is unreliable, a local Llama 3.2 instance is queried using a base64 prompt.

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
git clone --recursive git@github.com:EdwinTSalcedo/BLPR.git bplr
cd bplr
```

**Option 1: Using requirements.txt**

```bash
conda create -n bplr python=3.11
conda activate bplr
pip install -r requirements.txt
```

**Option 2: Using pyproject.toml**

```bash
conda create -n bplr python=3.11
conda activate bplr
pip install -e .
```

## Inference

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

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{auza_banegas2026,
  title   = {Robust License Plate Recognition Under Viewpoint and Illumination Variations: A Bolivian Case Study},
  author  = {Auza Banegas, Guillermo J. and Castro Sandoval, Sergio S. and Calvimontes Vera, Diego A. and Condori Peredo, Natalia L. and Salcedo, Edwin},
  journal = {IEEE Latin America Transactions}
}
```
