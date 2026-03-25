# Robust License Plate Recognition Under Viewpoint and Illumination Variations: A Bolivian Case Study
This repository contains the official implementation of the Bolivian license plate recognition (BLPR) framework, designed to operate robustly under perspective distortions and extreme illumination variations. The system uses a two-stage pipeline based on YOLOv11 and YOLOv8x, with an innovative fallback mechanism powered by Llama 3.2.

## 🚀 Key Features

- **Two-Stage Detection:** Uses YOLOv11n for vehicle/license plate localization and YOLOv8x for optical character recognition (OCR).


- **Perspective Rectification:** A preprocessing algorithm that corrects tilted or distorted plates using perspective transformations and Canny edge detection.


- **Illumination Enhancement:** Dynamic luminance adjustment (CLAHE for low light and gamma correction for overexposure) before recognition.


- **LLM Fallback:** Llama 3.2 integration to infer characters in cases of low confidence (confidence < 0.5) or inconsistent formats.


- **BLPR Dataset:** Introduction of the first specialized dataset for Bolivian license plates, including more than 49,000 synthetic images and real samples from La Paz.

## 🏗️ System Architecture
The pipeline processes images as follows:


1. **Detection:** YOLOv11n locates the vehicle and the license plate.

2. **Preprocessing:** The system validates that the plate is within the vehicle and extracts the ROI with a 10-pixel margin.

3. **Illumination Normalization:** The ROI is converted to YUV space to evaluate average luminance and apply enhancements based on the threshold.

4. **Rectification:** The image is binarized and edges are detected to apply a perspective transformation.

5. **Primary OCR:** YOLOv8x performs character recognition.

6. **Validation and Fallback:** If OCR fails or is unreliable, a local Llama 3.2 instance is queried using a base64 prompt.

## 📊 Performance
The system achieves 88.4% character-level accuracy on real street images under uncontrolled conditions.

| Model    | Task                                      | IoU    | F1-Score | Inference Time       |
|----------|-------------------------------------------|--------|----------|----------------------|
| YOLOv11n | Detection (LPD)                           | 0.7639 | 0.8682   | 3.62 ms              |
| YOLOv8x  | Optical character recognition (LPR)       | 0.9458 | 0.9837   | 29.74 ms             |

## 📂 BLPR Datasets
Four sub-datasets were developed for this study:
| Dataset | Samples  | Type      | Description                                                                          |
|---------|----------|-----------|--------------------------------------------------------------------------------------|
| BLPR-A  | 3,550    | Real      | Street photos from La Paz with vehicle and license plate annotations.                |
| BLPR-B  | 49,392   | Synthetic | Created in Blender with weather variations and angles from 30 degrees to 150 degrees.|
| BLPR-C  | 1,374    | Real/Aug  | Individual plates for OCR training with class balancing.                             |
| BLPR-D  | 499      | Real      | Validation dataset with luminance (lux) data and real perspective conditions.        |

## 📝 References and Authors
This work was presented in IEEE Latin America Transactions.

* **Authors:** Guillermo J. Auza Banegas, Sergio S. Castro Sandoval, Diego A. Calvimontes Vera, Natalia L. Condori Peredo, and Edwin Salcedo.

* **Institution:** Universidad Católica Boliviana "San Pablo" and Queen Mary University of London.
