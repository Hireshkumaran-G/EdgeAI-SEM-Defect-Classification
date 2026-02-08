import os
import cv2
import json
import numpy as np
import onnxruntime as ort
from glob import glob

# ===================== PATH SETUP =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

ONNX_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "mobilenetv3_sem.onnx"
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT, "demo_inputs", "images"
)

OUTPUT_VIDEO_PATH = os.path.join(
    PROJECT_ROOT, "outputs", "inference_demo_output.mp4"
)

CLASS_MAP_PATH = os.path.join(
    PROJECT_ROOT,"src", "class_map.json"
)

os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

# ===================== CONFIG =====================
IMG_SIZE = 128              # MUST match training
DISPLAY_SCALE = 2.0         # ONLY for visualization
SLIDESHOW_FPS = 1           # slow & readable

# ===================== LOAD CLASS MAP =====================
with open(CLASS_MAP_PATH, "r") as f:
    class_map = json.load(f)

CLASS_NAMES = [k for k, v in sorted(class_map.items(), key=lambda x: x[1])]

# ===================== LOAD ONNX MODEL =====================
session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name

# ===================== PREPROCESS (MATCH TRAINING) =====================
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.stack([img, img, img], axis=-1)  # 3-channel grayscale
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))        # CHW
    img = np.expand_dims(img, axis=0)
    return img

# ===================== LOAD IMAGES =====================
image_paths = sorted(
    glob(os.path.join(IMAGE_DIR, "*.png")) +
    glob(os.path.join(IMAGE_DIR, "*.jpg")) +
    glob(os.path.join(IMAGE_DIR, "*.jpeg"))
)

if len(image_paths) == 0:
    raise RuntimeError("No images found in demo_inputs/images")

# ===================== PREPARE VIDEO WRITER =====================
first_img = cv2.imread(image_paths[0])
first_img = cv2.resize(
    first_img,
    None,
    fx=DISPLAY_SCALE,
    fy=DISPLAY_SCALE,
    interpolation=cv2.INTER_CUBIC
)

h, w, _ = first_img.shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    fourcc,
    SLIDESHOW_FPS,
    (w, h)
)

print("▶ Running slideshow inference...")

# ===================== INFERENCE LOOP =====================
for img_path in image_paths:
    frame = cv2.imread(img_path)

    # Resize for DISPLAY ONLY
    frame = cv2.resize(
        frame,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_CUBIC
    )

    # Inference uses ORIGINAL frame
    inp = preprocess(cv2.imread(img_path))
    probs = session.run(None, {input_name: inp})[0][0]

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    label = f"{CLASS_NAMES[pred_idx]}"

    cv2.putText(
        frame,
        label,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )

    out.write(frame)

out.release()

print(f"Inference demo video saved at:\n{OUTPUT_VIDEO_PATH}")
