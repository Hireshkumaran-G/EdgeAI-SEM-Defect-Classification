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

BASE_DIR    = os.getcwd()
TFLITE_PATH = os.path.join(BASE_DIR, "outputs", "tflite", "model_dynamic.tflite")
TEST_DIR    = os.path.join(BASE_DIR, "Hackathon_phase3_prediction_dataset")
OUT_FILE    = os.path.join(BASE_DIR, "outputs", "predictions.txt")
IMG_SIZE    = (128, 128)
CLASSES = [
    "BRIDGE","CLEAN_CRACK","CLEAN_LAYER","CLEAN_VIA",
    "CMP","CRACK","LER","OPEN","OTHERS","PARTICLE","VIA"
]

interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]

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
    print(f"\n[ERROR] No images found in: {TEST_DIR}")
    exit(1)

print(f"Images : {len(image_paths)} found\n")

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
        print(f"  Processed {i+1}/{len(image_paths)} ...")

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w") as f:
    f.write("image_name,predicted_class\n")
    for name, cls in results:
        f.write(f"{name},{cls}\n")

print(f"\nSaved  : {OUT_FILE}  ({len(results)} rows)")

print("\nPreview (first 10):")
print(f"{'image_name':<45} predicted_class")
print("-" * 60)
for name, cls in results[:10]:
    print(f"  {name:<43} {cls}")
