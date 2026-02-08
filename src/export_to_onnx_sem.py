import os
import torch
import torch.nn as nn
from torchvision import models

# ================= CONFIG =================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODELPATH = os.path.join(MODEL_DIR, "mobilenetv3_sem_best.pth")
ONNXPath = os.path.join(MODEL_DIR, "mobilenetv3_sem.onnx")


IMG_SIZE = 128
NUM_CLASSES = 10
DEVICE = "cpu"   # ONNX export should always use CPU

os.makedirs(MODEL_DIR, exist_ok=True)

# ================= MODEL =================
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)

# Load weights safely
state_dict = torch.load(MODELPATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)

model.eval()
model.to(DEVICE)

# ================= DUMMY INPUT =================
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

# ================= EXPORT =================
torch.onnx.export(
    model,
    dummy_input,
    ONNXPath,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=None   # STATIC shape (important for edge)
)

print("✅ ONNX model exported successfully")
print("Saved at:", ONNXPath)
