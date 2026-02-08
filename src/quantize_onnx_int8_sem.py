import os
import numpy as np
from PIL import Image
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType
)

# ================= CONFIG =================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
# NOTE:
# Dataset is expected at <project_root>/dataset and is not included in GitHub.
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

ONNX_FP32_PATH = os.path.join(MODEL_DIR, "mobilenetv3_sem.onnx")
ONNX_INT8_PATH = os.path.join(MODEL_DIR, "mobilenetv3_sem_int8.onnx")
CALIB_DIR = os.path.join(DATA_DIR, "train")

IMG_SIZE = 128
NUM_CALIB_IMAGES = 200

# ================= CALIBRATION DATA READER =================
class SEMCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_dir):
        self.image_paths = []
        for cls in os.listdir(image_dir):
            cls_dir = os.path.join(image_dir, cls)
            if os.path.isdir(cls_dir):
                for f in os.listdir(cls_dir):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.image_paths.append(os.path.join(cls_dir, f))

        np.random.shuffle(self.image_paths)
        self.image_paths = self.image_paths[:NUM_CALIB_IMAGES]
        self.index = 0

    def preprocess(self, img_path):
        img = Image.open(img_path).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img).astype(np.float32) / 255.0

        # repeat channel → 3
        img = np.stack([img, img, img], axis=0)
        img = np.expand_dims(img, axis=0)
        return img

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        img = self.preprocess(self.image_paths[self.index])
        self.index += 1
        return {"input": img}

# ================= QUANTIZATION =================
print("🔧 Starting INT8 quantization...")
print("Calibrating using real SEM images...")

calib_reader = SEMCalibrationDataReader(CALIB_DIR)

quantize_static(
    model_input=ONNX_FP32_PATH,
    model_output=ONNX_INT8_PATH,
    calibration_data_reader=calib_reader,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8
)

print("✅ INT8 quantization completed")
print("INT8 model saved at:", ONNX_INT8_PATH)

# ================= SIZE COMPARISON =================
fp32_size = os.path.getsize(ONNX_FP32_PATH) / (1024 * 1024)
int8_size = os.path.getsize(ONNX_INT8_PATH) / (1024 * 1024)

print("\n📦 Model size comparison:")
print(f"FP32 : {fp32_size:.2f} MB")
print(f"INT8 : {int8_size:.2f} MB")
print(f"Compression ratio: {fp32_size / int8_size:.1f}×")
