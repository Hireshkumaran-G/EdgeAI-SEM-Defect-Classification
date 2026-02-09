# SEM Defect Classification – Edge AI

This repository contains an end-to-end Edge AI pipeline for classifying semiconductor wafer defects from Scanning Electron Microscope (SEM) images. The project demonstrates how deep learning can be optimized for edge deployment while maintaining high accuracy.

## Problem Statement
Manual inspection of SEM images is time-consuming and often prone to human error.  
This project automates the process using a lightweight convolutional neural network trained and quantized for efficient on-device (edge) inference.

## Defect Classes
- clean
- defect bridge
- defect cmp
- defect crack
- defect ler
- defect open
- defect particle
- defect scratch
- defect vias
- other

## Dataset Structure
```text
    dataset/
    ├── train/    (each defect class 150 images)
    ├── val/      (each defect class 32 images)
    └── test/     (each defect class 33 images)
```

Each split contains class-wise subfolders representing the defect categories.

Note: The full dataset is not included in this repository due to size constraints.

## Model Architecture
- Backbone: MobileNetV3-Small
- Input size: 128×128
- Training method: Transfer learning with selective fine-tuning

## Training and Evaluation
1. Train the model
   ```bash
    python src/train_mobilenetv3_sem.py
   ```
2. Evaluate model on test set
    ```bash
   python src/evaluate_test_sem_and_save.py
   ```
3. Export model to ONNX format
   ```bash
   python src/export_to_onnx_sem.py
   ```
4. Perform INT8 quantization (Edge optimization)
   ```bash
   python src/quantize_onnx_int8_sem.py
   ```
## Model Performance
Metric              | FP32     | INT8
--------------------|----------|--------
Model size          | 5.84 MB  | 1.67 MB (≈3.5× smaller)
Test Accuracy       | 96.37%   | Hardware-dependent (NXP eIQ)


- Balanced performance across all defect categories.
- Confusion matrix and output samples are available in outputs_sample/.

## Inference Demo
A headless ONNX-based inference demo (image slideshow / video) is provided to demonstrate
end-to-end SEM defect classification suitable for edge deployment.

## Deployment Compatibility
- Export format: ONNX
- Optimization: INT8 quantized
- Target platform: NXP Edge platforms (deployment via NXP eIQ toolkit)
- Runtime: ONNX Runtime (CPU)

## Hardware & Platform
Stage        | Framework    | Hardware
-------------|--------------|------------
Training     | PyTorch      | GPU (rtx 3050 4gb vram)
Inference    | ONNX Runtime | CPU / Edge device

## Disclaimer
The INT8 accuracy may vary depending on the target hardware and runtime calibration.  
Validation should be performed on the deployed NXP device using the eIQ runtime environment.

---
