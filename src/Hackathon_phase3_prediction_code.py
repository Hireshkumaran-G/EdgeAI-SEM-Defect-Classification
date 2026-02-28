"""
Inference using dynamic range TFLite model ONLY.
Output: predictions.txt → image_name,predicted_class
"""
import os, glob, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime

BASE_DIR    = os.getcwd()
TFLITE_PATH = os.path.join(BASE_DIR, "outputs", "tflite", "model_dynamic.tflite")
TEST_DIR    = os.path.join(BASE_DIR, "Hackathon_phase3_prediction_dataset")
OUT_FILE    = os.path.join(BASE_DIR, "outputs", "predictions.txt")
IMG_SIZE    = (128, 128)
CLASSES = [
    "BRIDGE","CLEAN_CRACK","CLEAN_LAYER","CLEAN_VIA",
    "CMP","CRACK","LER","OPEN","OTHERS","PARTICLE","VIA"
]

LOG_FILE = os.path.join(BASE_DIR, "outputs", "prediction_log.txt")

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as lf:
        lf.write(str(msg) + "\n")

# Clear previous log
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
open(LOG_FILE, "w").close()

# PHASE START LOGGING
log("========== PHASE 3 EVALUATION START ==========")
log(f"Date: {datetime.now()}")
log(f"Model Path: {TFLITE_PATH}")
log(f"Dataset Path: {TEST_DIR}")
log(f"Image Size: {IMG_SIZE[0]}")
log("No retraining performed as per rules.\n")

# Model Loaded Successfully
log("Model Loaded Successfully")
interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
log(f"Input Shape: {inp_d['shape']}")
log(f"Detected Test Classes: {CLASSES}\n")

assert inp_d["dtype"] == np.float32, \
    f"Expected float32 input for dynamic model, got {inp_d['dtype']}"

print(f"Model  : {TFLITE_PATH}")
print(f"Input  : dtype={inp_d['dtype']}  shape={inp_d['shape']}")
print(f"Norm   : raw [0..255] passed directly (MobileNetV3 rescales internally)")

image_paths = sorted(
    glob.glob(os.path.join(TEST_DIR, "**", "*.png"),  recursive=True) +
    glob.glob(os.path.join(TEST_DIR, "**", "*.jpg"),  recursive=True) +
    glob.glob(os.path.join(TEST_DIR, "**", "*.jpeg"), recursive=True) +
    glob.glob(os.path.join(TEST_DIR, "**", "*.bmp"),  recursive=True)
)

if not image_paths:
    log(f"\n[ERROR] No images found in: {TEST_DIR}")
    exit(1)

log(f"Images : {len(image_paths)} found\n")

results, gt, pd = [], [], []

for i, path in enumerate(image_paths):
    img = np.array(
        Image.open(path).convert("RGB").resize(IMG_SIZE, Image.BILINEAR),
        dtype=np.float32
    )
    interp.set_tensor(inp_d["index"], img[np.newaxis])
    interp.invoke()
    pred_idx = int(np.argmax(interp.get_tensor(out_d["index"])))
    results.append((os.path.basename(path), CLASSES[pred_idx]))
    pd.append(pred_idx)
    parent = os.path.basename(os.path.dirname(path))
    if parent in CLASSES:
        gt.append(CLASSES.index(parent))
    if (i + 1) % 100 == 0:
        log(f"  Processed {i+1}/{len(image_paths)} ...")

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w") as f:
    f.write("image_name,predicted_class\n")
    for name, cls in results:
        f.write(f"{name},{cls}\n")

log(f"\nSaved  : {OUT_FILE}  ({len(results)} rows)\n")

# Only preview and phase end

log("\nPreview (first 10):")
log(f"{'image_name':<45} predicted_class")
log("-" * 60)
for name, cls in results[:10]:
    log(f"  {name:<43} {cls}")

log("\n========== PHASE 3 EVALUATION COMPLETE ==========")
