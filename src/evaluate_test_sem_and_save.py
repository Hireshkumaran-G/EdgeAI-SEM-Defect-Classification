import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ================= CONFIG =================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
# NOTE:
# Dataset is expected at <project_root>/dataset and is not included in GitHub.
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


IMG_SIZE = 128
BATCH_SIZE = 16
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= OUTPUT DIR =================
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Results will be saved to:", OUTPUT_DIR)

# ================= TRANSFORMS =================
test_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ================= DATASET =================
test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_tf)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class_names = test_ds.classes
print("Class order:", class_names)
print("Test images:", len(test_ds))

# ================= MODEL =================
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "mobilenetv3_sem_best.pth"), map_location=DEVICE, weights_only=True))
model = model.to(DEVICE)
model.eval()

# ================= INFERENCE =================
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        preds = torch.argmax(out, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ================= METRICS =================
test_acc = (all_preds == all_labels).mean()

report = classification_report(
    all_labels,
    all_preds,
    target_names=class_names,
    digits=4
)

cm = confusion_matrix(all_labels, all_preds)

# ================= SAVE TEXT RESULTS =================
with open(os.path.join(OUTPUT_DIR, "test_accuracy.txt"), "w") as f:
    f.write(f"Test Accuracy: {test_acc:.4f}\n")

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

np.savetxt(
    os.path.join(OUTPUT_DIR, "confusion_matrix.csv"),
    cm,
    delimiter=",",
    fmt="%d"
)

print(f"\n✅ TEST ACCURACY: {test_acc:.4f}")
print("Reports saved.")

# ================= CONFUSION MATRIX PLOT =================
plt.figure(figsize=(14, 12))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    cbar=True,
    annot_kws={"size": 12}
)

plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.title("SEM Defect Classification – Confusion Matrix", fontsize=16)

plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(rotation=0, fontsize=12)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=300
)
plt.close()

print("📊 Confusion matrix plot saved.")
