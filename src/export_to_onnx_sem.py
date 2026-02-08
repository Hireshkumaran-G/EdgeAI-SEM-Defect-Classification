import os
import torch
import torch.nn as nn
from torchvision import models

# ================= CONFIG =================
MODEL_PATH = "models/mobilenetv3_sem_best.pth"
OUTPUT_DIR = os.path.join("outputs")
ONNX_PATH = os.path.join(OUTPUT_DIR, "mobilenetv3_sem.onnx")

IMG_SIZE = 128
NUM_CLASSES = 10
DEVICE = "cpu"   # ONNX export should always use CPU

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= MODEL =================
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)

# Load weights safely
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)

model.eval()
model.to(DEVICE)

# ================= DUMMY INPUT =================
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

# ================= EXPORT =================
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=None   # STATIC shape (important for edge)
)

print("✅ ONNX model exported successfully")
print("Saved at:", ONNX_PATH)
