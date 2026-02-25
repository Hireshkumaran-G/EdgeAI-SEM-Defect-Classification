"""
Produces: accuracy, precision, recall, F1, confusion matrix,
          model sizes (float32 vs quantized).
"""
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)
import seaborn as sns

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR     = os.getcwd()  # base directory for reproducibility
DATASET_DIR  = os.path.join(BASE_DIR, "Hackathon_phase3_training_dataset_augmented")
CKPT_PATH    = os.path.join(BASE_DIR, "outputs", "checkpoints", "best_p3_finetune.h5")
TFLITE_DYN   = os.path.join(BASE_DIR, "outputs", "tflite", "model_dynamic.tflite")
TFLITE_FB    = os.path.join(BASE_DIR, "outputs", "tflite", "model_int8_fallback.tflite")
TFLITE_STR   = os.path.join(BASE_DIR, "outputs", "tflite", "model_int8_strict.tflite")
SAVED_MODEL  = os.path.join(BASE_DIR, "outputs", "models", "saved_model_float32")
OUT_DIR      = os.path.join(BASE_DIR, "outputs", "evaluation")
IMG_SIZE     = (128, 128)
BATCH_SIZE   = 32
VAL_SPLIT    = 0.2
SEED         = 42
CLASSES = [
    "BRIDGE","CLEAN_CRACK","CLEAN_LAYER","CLEAN_VIA",
    "CMP","CRACK","LER","OPEN","OTHERS","PARTICLE","VIA"
]

os.makedirs(OUT_DIR, exist_ok=True)  # ensure output directory exists

# ── Helpers ──────────────────────────────────────────────────────────────────
def make_val_generator(batch_size=BATCH_SIZE):
    gen = ImageDataGenerator(validation_split=VAL_SPLIT)
    return gen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, color_mode="rgb",
        classes=CLASSES, class_mode="categorical",
        batch_size=batch_size, subset="validation",
        seed=SEED, shuffle=False,
    )

def get_dir_size_mb(path):
    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(path) for f in fs)
    return total / 1024 / 1024

def print_metrics(y_true, y_pred, title):
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    p   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r   = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {p*100:.2f}%  (macro)")
    print(f"  Recall    : {r*100:.2f}%  (macro)")
    print(f"  F1 Score  : {f1*100:.2f}%  (macro)")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)}")
    return acc, p, r, f1

def save_confusion_matrix(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Confusion matrix saved: {path}")

def eval_tflite(tflite_path):
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    scale_in, zp_in = inp["quantization"]
    val_ds = make_val_generator(batch_size=1)
    val_ds.reset()
    y_pred = []
    for imgs, _ in val_ds:
        x = imgs[0][np.newaxis].astype(np.float32)
        if inp["dtype"] == np.int8:
            x = np.round(x / scale_in + zp_in).clip(-128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        pred = int(np.argmax(interp.get_tensor(out["index"])))  # fixed: removed extra )
        y_pred.append(pred)
        if len(y_pred) >= val_ds.samples:
            break
    return y_pred[:val_ds.samples]

# ── 1. Float32 ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("LOADING FLOAT32 MODEL")
print("="*60)
model = keras.models.load_model(CKPT_PATH, compile=False)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

val_ds = make_val_generator()
val_ds.reset()
y_true, y_pred_f32 = [], []
for imgs, labels in val_ds:
    preds = model.predict(imgs, verbose=0)
    y_pred_f32.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels, axis=1))
    if len(y_true) >= val_ds.samples:
        break
y_true     = y_true[:val_ds.samples]
y_pred_f32 = y_pred_f32[:val_ds.samples]

acc_f32, p_f32, r_f32, f1_f32 = print_metrics(y_true, y_pred_f32, "Float32 Model")
save_metrics_txt(acc_f32, p_f32, r_f32, "metrics_float32.txt")
save_confusion_matrix(y_true, y_pred_f32, "Float32 Confusion Matrix", "confusion_matrix_float32.png")

# ── 2. Dynamic TFLite ─────────────────────────────────────────────────────────
print("\nEvaluating Dynamic TFLite ...")
y_pred_dyn = eval_tflite(TFLITE_DYN)
acc_dyn, p_dyn, r_dyn, f1_dyn = print_metrics(y_true, y_pred_dyn, "Dynamic Range TFLite (PRIMARY)")
save_metrics_txt(acc_dyn, p_dyn, r_dyn, "metrics_dynamic.txt")
save_confusion_matrix(y_true, y_pred_dyn, "Dynamic TFLite Confusion Matrix", "confusion_matrix_dynamic.png")

# ── 3. INT8 Fallback TFLite ───────────────────────────────────────────────────
print("\nEvaluating INT8 Fallback TFLite ...")
y_pred_fb = eval_tflite(TFLITE_FB)
acc_fb, p_fb, r_fb, f1_fb = print_metrics(y_true, y_pred_fb, "INT8 Fallback TFLite (BACKUP)")
save_metrics_txt(acc_fb, p_fb, r_fb, "metrics_int8_fallback.txt")
save_confusion_matrix(y_true, y_pred_fb, "INT8 Fallback Confusion Matrix", "confusion_matrix_int8_fallback.png")

# ── 4. Strict INT8 TFLite ─────────────────────────────────────────────────────
print("\nEvaluating Strict INT8 TFLite ...")
y_pred_str = eval_tflite(TFLITE_STR)
acc_str, p_str, r_str, f1_str = print_metrics(y_true, y_pred_str, "Strict INT8 TFLite (MCU BACKUP)")
save_metrics_txt(acc_str, p_str, r_str, "metrics_int8_strict.txt")
save_confusion_matrix(y_true, y_pred_str, "Strict INT8 Confusion Matrix", "confusion_matrix_int8_strict.png")

# ── 5. Model sizes ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL SIZES")
print("="*60)
sm_size   = get_dir_size_mb(SAVED_MODEL) if os.path.exists(SAVED_MODEL) else 0
h5_size   = os.path.getsize(CKPT_PATH)   / 1024 / 1024 if os.path.exists(CKPT_PATH) else 0
dyn_size  = os.path.getsize(TFLITE_DYN)  / 1024 if os.path.exists(TFLITE_DYN) else 0
fb_size   = os.path.getsize(TFLITE_FB)   / 1024 if os.path.exists(TFLITE_FB)  else 0
str_size  = os.path.getsize(TFLITE_STR)  / 1024 if os.path.exists(TFLITE_STR) else 0

print(f"  Float32 H5 checkpoint      : {h5_size:.2f} MB")
print(f"  Float32 SavedModel         : {sm_size:.2f} MB")
print(f"  Dynamic TFLite   (PRIMARY) : {dyn_size:.1f} KB")
print(f"  INT8 Fallback    (BACKUP)  : {fb_size:.1f} KB")
print(f"  Strict INT8      (MCU)     : {str_size:.1f} KB")
if sm_size > 0 and dyn_size > 0:
    print(f"  Size reduction (SM→Dynamic): {(1-(dyn_size/1024)/sm_size)*100:.1f}%")
if sm_size > 0 and str_size > 0:
    print(f"  Size reduction (SM→Strict) : {(1-(str_size/1024)/sm_size)*100:.1f}%")

# ── 6. Summary table ──────────────────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Model':<35} {'Accuracy':>9} {'Precision':>9} {'Recall':>9} {'F1':>9} {'Size':>10}")
print("-"*80)
print(f"{'Float32 (baseline)':<35} {acc_f32*100:>8.2f}% {p_f32*100:>8.2f}% {r_f32*100:>8.2f}% {f1_f32*100:>8.2f}% {h5_size:>7.2f} MB")
print(f"{'Dynamic TFLite  ★ PRIMARY':<35} {acc_dyn*100:>8.2f}% {p_dyn*100:>8.2f}% {r_dyn*100:>8.2f}% {f1_dyn*100:>8.2f}% {dyn_size:>7.1f} KB")
print(f"{'INT8 Fallback   BACKUP':<35} {acc_fb*100:>8.2f}%  {p_fb*100:>8.2f}% {r_fb*100:>8.2f}% {f1_fb*100:>8.2f}% {fb_size:>7.1f} KB")
print(f"{'Strict INT8     MCU BACKUP':<35} {acc_str*100:>8.2f}%  {p_str*100:>8.2f}% {r_str*100:>8.2f}% {f1_str*100:>8.2f}% {str_size:>7.1f} KB")

print(f"\nAll outputs saved to: {OUT_DIR}/")
print("  confusion_matrix_float32.png")
print("  confusion_matrix_dynamic.png")
print("  confusion_matrix_int8_fallback.png")
print("  confusion_matrix_int8_strict.png")
