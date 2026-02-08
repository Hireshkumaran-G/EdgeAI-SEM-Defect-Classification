import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ===================== BASIC SETUP =====================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
# NOTE:
# Dataset is expected at <project_root>/dataset and is not included in GitHub.
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ===================== CONFIG =====================
NUM_CLASSES = 10
IMG_SIZE = 128
BATCH_SIZE = 24
EPOCHS = 45
LR = 3e-4
FREEZE_EPOCHS = 3

# ===================== TRANSFORMS =====================
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ===================== DATASETS =====================
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_tf)

print("Class mapping:", train_ds.class_to_idx)
print("Train images:", len(train_ds))
print("Val images:", len(val_ds))

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0   # WINDOWS SAFE
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ===================== MODEL =====================
model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ===================== FREEZE BACKBONE =====================
for param in model.features.parameters():
    param.requires_grad = False

print("Backbone frozen for first", FREEZE_EPOCHS, "epochs")

# ===================== LOSS & OPTIMIZER =====================
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ===================== TRAINING LOOP =====================
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # ---- Unfreeze backbone after warmup ----
    if epoch == FREEZE_EPOCHS:
        print("Unfreezing backbone...")
        for param in model.features.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---- TRAIN ----
    model.train()
    train_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()

    # ---- VALIDATION ----
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = correct / total
    avg_train_loss = train_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    # ---- SAVE BEST MODEL ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "mobilenetv3_sem_best.pth"))
        print("Best model saved")

print("\nTraining complete.")
print("Best validation accuracy:", best_val_acc)
