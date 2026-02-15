import os
import cv2
import json
import sys
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from datetime import datetime

# ================= PATH SETUP =================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mobilenetv3_sem.onnx")
DATASET_PATH = os.path.join(PROJECT_ROOT, "hackathon_test_dataset")
CLASS_MAP_PATH = os.path.join(PROJECT_ROOT, "src", "class_map.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "phase2")

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 128

# ================= LOG FILE SETUP =================
log_path = os.path.join(OUTPUT_DIR, "prediction_log.txt")
log_file = open(log_path, "w")

class Logger:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Logger(sys.stdout, log_file)
sys.stderr = Logger(sys.stderr, log_file)

print("========== PHASE 2 EVALUATION START ==========")
print("Date:", datetime.now())
print("Model Path:", MODEL_PATH)
print("Dataset Path:", DATASET_PATH)
print("Image Size:", IMG_SIZE)
print("No retraining performed as per rules.\n")

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
        print("Warning: Could not read", img_path)
        return None

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

print("\n========== RESULTS ==========")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))

print("\nConfusion Matrix:")
print("Labels:", TEST_CLASSES)
print(cm)

print("\nClassification Report:\n")
print(report)

# ================= SAVE RESULTS =================

# Save metrics
with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision (weighted): {precision:.4f}\n")
    f.write(f"Recall (weighted): {recall:.4f}\n")

# Save classification report
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Save Confusion Matrix Image
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

print("========== PHASE 2 EVALUATION COMPLETE ==========")
