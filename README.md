# Robust License Plate Recognition Under Viewpoint and Illumination Variations: A Bolivian Case Study
Este repositorio contiene la implementación oficial del framework de reconocimiento de placas bolivianas (BLPR), diseñado para operar de manera robusta ante distorsiones de perspectiva y variaciones extremas de iluminación. El sistema utiliza un pipeline de dos etapas basado en YOLOv11 y YOLOv8x, con un innovador mecanismo de respaldo (fallback) mediante Llama 3.2.

## 🚀 Características Principales

- **Detección de Dos Etapas:** Utiliza YOLOv11n para la localización de vehículos/placas y YOLOv8x para el reconocimiento de caracteres (OCR).


- **Rectificación de Perspectiva:** Algoritmo de preprocesamiento que corrige placas inclinadas o deformadas mediante transformaciones de perspectiva y detección de bordes Canny.


- **Mejora de Iluminación:** Ajuste dinámico de luminancia (CLAHE para baja luz y corrección gamma para sobreexposición) antes del reconocimiento.


- **LLM Fallback:** Integración de Llama 3.2 para inferir caracteres en casos de baja confianza (confianza < 0.5) o formatos inconsistentes.


- **Dataset BLPR:** Introducción del primer dataset especializado en placas bolivianas, incluyendo más de 49,000 imágenes sintéticas y muestras reales de La Paz.

## 🏗️ Arquitectura del Sistema
El pipeline procesa las imágenes de la siguiente manera:


1. **Detección:** YOLOv11n localiza el vehículo y la placa.

2. **Preprocesamiento:** Se valida que la placa esté dentro del vehículo y se extrae el ROI con un margen de 10 píxeles.

3. **Normalización de Iluminación:** El ROI se convierte a espacio YUV para evaluar la luminancia media y aplicar mejoras según el umbral.

4. **Rectificación:** Se binariza la imagen y se detectan los bordes para aplicar una transformación de perspectiva.

5. **OCR Primario:** YOLOv8x realiza el reconocimiento de caracteres.

6. **Validación y Fallback:** Si el OCR falla o es poco confiable, se consulta a una instancia local de Llama 3.2 mediante un prompt base64.

## 📊 Rendimiento
El sistema logra una precisión a nivel de caracteres del 88.4% en imágenes reales de calle bajo condiciones no controladas.

| Modelo   | Tarea                                     | IoU    | F1-Score | Tiempo de Inferencia |
|----------|-------------------------------------------|--------|----------|----------------------|
| YOLOv11n | Detección (LPD)                           | 0,7639 | 0.8682   | 3,62 ms              |
| Yolov8x  | Reconocimiento óptico de caracteres (LPR) | 0.9458 | 0.9837   | 29,74 ms             |

## 📂 Datasets BLPR
Se desarrollaron cuatro sub-datasets para este estudio:
| Dataset | Muestras | Tipo      | Descripción                                                                          |
|---------|----------|-----------|--------------------------------------------------------------------------------------|
| BLPR-A  | 3,550    | Real      | Fotos de calles en La Paz con anotaciones de vehículos y placas.                     |
| BLPR-B  | 49.392   | Sintético | Creado en Blender con variaciones de clima y ángulos de 30° a 150°.                  |
| BLPR-C  | 1.374    | Real/Aug  | Placas individuales para entrenamiento de OCR con balanceo de clases.                |
| BLPR-D  | 499      | Real      | Conjunto de datos de validación con datos de luminancia (lux) y perspectivas reales. |

## 📝 Referencias y Autores
Este trabajo fue presentado en IEEE Latin America Transactions.

* **Autores:** Guillermo J. Auza Banegas, Sergio S. Castro Sandoval, Diego A. Calvimontes Vera, Natalia L. Condori Peredo y Edwin Salcedo.

* **Institución:** Universidad Católica Boliviana "San Pablo" y Queen Mary University of London.