import os
import torch
import torch.nn as nn
from torchvision import models

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "mobilenetv3_sem_best.pth"
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ONNX_PATH = os.path.join(PROJECT_ROOT, "models", "mobilenetv3_sem.onnx")

# ================= CONFIG =================
NUM_CLASSES = 10
IMG_SIZE = 128
DEVICE = "cpu"   # ONNX export should be CPU

# ================= LOAD MODEL =================
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    NUM_CLASSES
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model.eval()

model.to(DEVICE)

# ================= DUMMY INPUT =================
dummy_input = torch.randn(
    1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE
)

# ================= EXPORT =================
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=13
)

print("ONNX model exported correctly (EVAL MODE)")
print("Saved at:", ONNX_PATH)
