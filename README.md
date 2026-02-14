# SEM Defect Classification – Edge AI

This repository contains an end-to-end Edge AI pipeline for classifying semiconductor wafer defects from Scanning Electron Microscope (SEM) images. The project demonstrates how deep learning can be optimized for efficient edge deployment while maintaining strong classification performance.

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
├── train/    (each defect class ~150 images)
├── val/      (each defect class ~32 images)
└── test/     (each defect class ~33 images)
```

Each split contains class-wise subfolders representing the defect categories.

**Note:** The full dataset is not included in this repository due to size constraints.

## Model Architecture

- Backbone: MobileNetV3-Small
- Input size: 128×128
- Training method: Transfer learning with selective fine-tuning
- Inference format: ONNX

---

## Phase 1 – Training and Evaluation

### Training and Evaluation Steps

1. **Train the model**
```bash
   python src/train_mobilenetv3_sem.py
```

2. **Evaluate model on test set**
```bash
   python src/evaluate_test_sem_and_save.py
```

3. **Export model to ONNX format**
```bash
   python src/export_to_onnx_sem.py
```

4. **Perform INT8 quantization (Edge optimization)**
```bash
   python src/quantize_onnx_int8_sem.py
```

### Phase 1 Model Performance

| Metric | FP32 | INT8 |
|--------|------|------|
| Model size | 5.84 MB | 1.67 MB (≈3.5× smaller) |
| Test Accuracy | 96.37% | Hardware-dependent (NXP eIQ) |

- Balanced performance across all defect categories.
- Confusion matrix and output samples are available in `outputs/`.

### Inference Demo

A headless ONNX-based inference demo (image slideshow / video) is provided to demonstrate end-to-end SEM defect classification suitable for edge deployment.

---

## Phase 2 – Hackathon Evaluation

### Phase 2 Evaluation Dataset

For Phase 2, a separate `hackathon_test_dataset` was provided by the organizers.

The following rules were strictly followed:

* Model re-training was NOT performed.
* The original Phase 1 ONNX model was used without modification.
* No new ONNX file was submitted.
* Only preprocessing and inference were performed.
* Accuracy, Precision, Recall, and Confusion Matrix were generated.

### Phase 2 Inference and Evaluation

Run Phase 2 evaluation using:
```bash
python src/hackathon_test_dataset_prediction.py
```

This script performs:

1. Preprocessing of hackathon_test_dataset images
2. Loading the Phase 1 ONNX model
3. Inference/prediction
4. Generation of confusion matrix
5. Calculation of accuracy, precision, and recall
6. Saving of prediction log

### Phase 2 Model Performance

| Metric | FP32 ONNX |
|--------|-----------|
| Model size | 5.84 MB |
| Accuracy | 36.82% |
| Precision (weighted) | 39.04% |
| Recall (weighted) | 36.82% |

* Confusion matrix, classification report, and logs are available in `outputs/phase2/`.
* Phase 1 internal validation results are available in `outputs/phase1/`.

### Observations

A distribution (domain) shift was observed between the Phase 1 training dataset and the Phase 2 hackathon evaluation dataset.

As per competition rules:

* No retraining was performed.
* Defect classes not present in the Phase 2 dataset were naturally mapped to the "other" category as permitted by the evaluation guidelines.
* The evaluation reflects real cross-domain generalization performance.

---

## Deployment Compatibility

- Export format: ONNX
- Optimization: INT8 quantization available
- Target platform: NXP Edge platforms (deployment via NXP eIQ toolkit)
- Runtime: ONNX Runtime (CPU)

## Hardware & Platform

| Stage | Framework | Hardware |
|-------|-----------|----------|
| Training | PyTorch | GPU (RTX 3050, 4GB VRAM) |
| Inference | ONNX Runtime | CPU / Edge device |

## Disclaimer

**Phase 1:** The INT8 accuracy may vary depending on the target hardware and runtime calibration. Validation should be performed on the deployed NXP device using the eIQ runtime environment.

**Phase 2:** Phase 2 evaluation strictly used the original Phase 1 ONNX model without retraining, in compliance with hackathon rules. Performance may vary under different dataset distributions due to domain shift.

---