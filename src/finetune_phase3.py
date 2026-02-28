"""
Quick phase-3 fine-tune of existing float32 model — no full retraining.
Just ~10 epochs at very low LR to squeeze more float32 accuracy.
"""
import os, random, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

BASE_DIR    = os.getcwd()  # base directory for reproducibility
CKPT_PATH   = os.path.join(BASE_DIR, "outputs", "checkpoints", "best_p3_finetune.h5")
DATASET_DIR = os.path.join(BASE_DIR, "Hackathon_phase3_training_dataset_augmented")
TFLITE_DIR  = os.path.join(BASE_DIR, "outputs", "tflite")
MODEL_DIR   = os.path.join(BASE_DIR, "outputs", "models")
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
VAL_SPLIT   = 0.2
P3_LR       = 2e-6   
P3_EPOCHS   = 30     
CLASSES = [
    "BRIDGE","CLEAN_CRACK","CLEAN_LAYER","CLEAN_VIA",
    "CMP","CRACK","LER","OPEN","OTHERS","PARTICLE","VIA"
]

def make_generator(subset, augment=False):
    kwargs = dict(validation_split=VAL_SPLIT)
    if augment:
        kwargs.update(rotation_range=10, width_shift_range=0.1,
                      height_shift_range=0.1, zoom_range=0.1,
                      horizontal_flip=True, fill_mode="nearest")
    return ImageDataGenerator(**kwargs).flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, color_mode="rgb",
        classes=CLASSES, class_mode="categorical",
        batch_size=BATCH_SIZE, subset=subset, seed=SEED,
        shuffle=(subset=="training"),
    )

train_ds = make_generator("training",   augment=True)
val_ds   = make_generator("validation", augment=False)

if not os.path.exists(CKPT_PATH):
    CKPT_PATH = os.path.join(BASE_DIR, "outputs", "checkpoints", "best_phase2.h5")  # fallback checkpoint

print(f"Loading {CKPT_PATH} ...")
model = keras.models.load_model(CKPT_PATH, compile=False)

for layer in model.layers:
    layer.trainable = True
    if isinstance(layer, layers.BatchNormalization) and layer.name != "head_bn":
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(P3_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("Baseline:")
val_ds.reset()
model.evaluate(val_ds, verbose=1)

model.fit(
    train_ds, validation_data=val_ds,
    epochs=P3_EPOCHS,
    callbacks=[
        ModelCheckpoint(os.path.join(BASE_DIR, "outputs", "checkpoints", "best_p3_finetune.h5"),
                        monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=6,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                          patience=3, min_lr=1e-8, verbose=1),
    ],
    verbose=1,
)

print("\nExporting and quantizing improved model ...")
model.save(os.path.join(MODEL_DIR, "saved_model_float32"), save_format="tf")  # save improved model

import shutil
shutil.copy(os.path.join(BASE_DIR, "outputs", "checkpoints", "best_p3_finetune.h5"),
            os.path.join(BASE_DIR, "outputs", "checkpoints", "best_phase3.h5"))  # backup best checkpoint

def rep_dataset():
    cal = ImageDataGenerator(validation_split=VAL_SPLIT).flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, color_mode="rgb",
        classes=CLASSES, class_mode="categorical",
        batch_size=1, subset="validation", seed=SEED, shuffle=False,
    )
    cal.reset()
    for i, (imgs, _) in enumerate(cal):
        if i >= 200: return
        yield [imgs[0][np.newaxis].astype(np.float32)]

conv = tf.lite.TFLiteConverter.from_saved_model(
    os.path.join(MODEL_DIR, "saved_model_float32"))
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_dataset
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type  = tf.int8
conv.inference_output_type = tf.int8
tflite = conv.convert()
out = os.path.join(TFLITE_DIR, "model_int8_p3finetune.tflite")
open(out, "wb").write(tflite)
print(f"Saved {out}  ({len(tflite)/1024:.1f} KB)")
