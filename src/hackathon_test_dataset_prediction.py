import os
import cv2
import json
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from datetime import datetime

# ================= PATH SETUP =================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mobilenetv3_sem_int8.onnx")  # Phase-1 submitted model
DATASET_PATH = os.path.join(PROJECT_ROOT, "hackathon_test_dataset")
CLASS_MAP_PATH = os.path.join(PROJECT_ROOT, "src", "class_map.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 128

# ================= LOAD CLASS MAP =================
with open(CLASS_MAP_PATH, "r") as f:
    class_map = json.load(f)

idx_to_class = {v: k for k, v in class_map.items()}

# ================= LOAD MODEL =================
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

print("Model Loaded Successfully")
print("Input Shape:", session.get_inputs()[0].shape)

# ================= PREPROCESS FUNCTION =================
def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Match Phase-1 preprocessing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.stack([img, img, img], axis=-1)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img

# ================= GET TEST CLASSES =================
TEST_CLASSES = [f.lower() for f in os.listdir(DATASET_PATH)]
print("Detected Test Classes:", TEST_CLASSES)

y_true = []
y_pred = []

# ================= INFERENCE LOOP =================
for folder in os.listdir(DATASET_PATH):
    folder_lower = folder.lower()
    folder_path = os.path.join(DATASET_PATH, folder)

    if not os.path.isdir(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        inp = preprocess(img_path)
        if inp is None:
            continue

        output = session.run(None, {input_name: inp})
        logits = output[0][0]

        pred_idx = int(np.argmax(logits))
        pred_class = idx_to_class[pred_idx]

        # If predicted class not in test dataset → classify as 'other'
        if pred_class not in TEST_CLASSES:
            pred_class = "other"

        y_true.append(folder_lower)
        y_pred.append(pred_class)

# ================= METRICS =================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=TEST_CLASSES)
report = classification_report(y_true, y_pred, zero_division=0)

# ================= SAVE RESULTS =================

# Accuracy / Precision / Recall
with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision (weighted): {precision:.4f}\n")
    f.write(f"Recall (weighted): {recall:.4f}\n")

# Classification report
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Confusion matrix CSV
np.savetxt(
    os.path.join(OUTPUT_DIR, "confusion_matrix.csv"),
    cm,
    delimiter=",",
    fmt="%d"
)

# Confusion matrix PNG
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=TEST_CLASSES,
    yticklabels=TEST_CLASSES
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Phase 2 Evaluation")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# Log file
with open(os.path.join(OUTPUT_DIR, "prediction_log.txt"), "w") as f:
    f.write("Phase 2 Evaluation Log\n")
    f.write("Date: " + str(datetime.now()) + "\n")
    f.write("Model Used: mobilenetv3_sem.onnx (Phase-1 submitted model)\n")
    f.write("Image Size: 128x128\n")
    f.write("No retraining performed as per rules.\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")

print("\n========== PHASE 2 EVALUATION COMPLETE ==========")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("All outputs saved in:", OUTPUT_DIR)
print("=================================================")
