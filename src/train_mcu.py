"""
MCU-Optimised Training Pipeline - IESA 2026 Phase-3
Target: NXP i.MX RT1170  |  Runtime: TFLite Micro INT8
Backbone: MobileNetV3Small alpha=0.75 input(128,128,3)
"""
import os, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR      = os.getcwd()  
DATASET_PATH  = os.path.join(BASE_DIR, "Hackathon_phase3_training_dataset_augmented")
OUTPUT_ROOT   = os.path.join(BASE_DIR, "outputs")
CKPT_DIR      = os.path.join(OUTPUT_ROOT, "checkpoints")
MODEL_DIR     = os.path.join(OUTPUT_ROOT, "models")
TFLITE_DIR    = os.path.join(OUTPUT_ROOT, "tflite")
LOG_DIR       = os.path.join(OUTPUT_ROOT, "logs")
for d in [CKPT_DIR, MODEL_DIR, TFLITE_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True) 

IMG_SIZE     = (128, 128)
BATCH_SIZE   = 32 
NUM_CLASSES  = 11
VAL_SPLIT    = 0.2
P1_EPOCHS    = 40 # Phase 1: Train head only
P1_LR        = 1e-4
P2_EPOCHS    = 30 # Phase 2: Unfreeze backbone, low LR
P2_LR        = 1e-5
DROPOUT_RATE = 0.3
L2_REG       = 1e-4

CLASSES = [
    "BRIDGE","CLEAN_CRACK","CLEAN_LAYER","CLEAN_VIA",
    "CMP","CRACK","LER","OPEN","OTHERS","PARTICLE","VIA"
]


def build_generators():
    train_gen = ImageDataGenerator(
        rotation_range=10, # degrees
        width_shift_range=0.10, # fraction of total width
        height_shift_range=0.10, # fraction of total height
        zoom_range=0.10, # fraction for zooming
        horizontal_flip=True, # randomly flip images
        fill_mode="nearest", # how to fill in new pixels after transformations
        validation_split=VAL_SPLIT,
    )
    val_gen = ImageDataGenerator(validation_split=VAL_SPLIT)
    train_ds = train_gen.flow_from_directory(
        DATASET_PATH, target_size=IMG_SIZE, color_mode="rgb",
        classes=CLASSES, class_mode="categorical",
        batch_size=BATCH_SIZE, subset="training", seed=SEED, shuffle=True,
    )
    val_ds = val_gen.flow_from_directory(
        DATASET_PATH, target_size=IMG_SIZE, color_mode="rgb",
        classes=CLASSES, class_mode="categorical",
        batch_size=BATCH_SIZE, subset="validation", seed=SEED, shuffle=False,
    )
    return train_ds, val_ds


def build_model():
    base = keras.applications.MobileNetV3Small(
        input_shape=(*IMG_SIZE, 3),
        alpha=0.75, # width multiplier for a smaller model
        include_top=False,
        weights="imagenet", # Use ImageNet weights as a starting point
        pooling="avg",
    )
    inputs = base.input
    x = base.output
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(NUM_CLASSES, kernel_regularizer=keras.regularizers.l2(L2_REG))(x)
    outputs = layers.Activation("softmax", dtype="float32", name="predictions")(x)
    return keras.Model(inputs, outputs, name="MobileNetV3Small_defect"), base


def freeze_backbone_bn(model): # Freeze all BatchNorm layers in the backbone except the head_bn
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization) and layer.name != "head_bn":
            layer.trainable = False


def set_phase1_trainability(model, base): # Freeze backbone, train head only
    base.trainable = False
    freeze_backbone_bn(model)


def set_phase2_trainability(model, base): # Unfreeze backbone, but keep BN frozen for stability
    base.trainable = True
    freeze_backbone_bn(model)


def get_callbacks(phase: int):
    tag = f"phase{phase}"
    return [
        ModelCheckpoint(
            filepath=os.path.join(CKPT_DIR, f"best_{tag}.h5"),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy", patience=12,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.3,
            patience=4, min_lr=1e-8, verbose=1,
        ),
        CSVLogger(os.path.join(LOG_DIR, f"{tag}_history.csv")),
    ]


def compute_class_weights(train_ds) -> dict:
    from sklearn.utils.class_weight import compute_class_weight
    labels = train_ds.classes
    unique = np.unique(labels)
    weights = compute_class_weight("balanced", classes=unique, y=labels)
    cw = {int(u): float(w) for u, w in zip(unique, weights)}
    print("Class weights:", {CLASSES[k]: f"{v:.2f}" for k, v in cw.items()})
    return cw


def train(model, base, train_ds, val_ds):
    class_weights = compute_class_weights(train_ds)

    # Phase 1 — head only
    print("\n" + "="*60)
    print("PHASE 1 - Head only, full backbone frozen")
    print("="*60)
    set_phase1_trainability(model, base)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=P1_LR),
        loss="categorical_crossentropy", metrics=["accuracy"],
    )
    model.summary(line_length=100)
    h1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=P1_EPOCHS, callbacks=get_callbacks(1),
        class_weight=class_weights, verbose=1,
    )
    best_p1 = os.path.join(CKPT_DIR, "best_phase1.h5")
    if os.path.exists(best_p1):
        model.load_weights(best_p1)
        print(f"Loaded best phase-1 weights from {best_p1}")

    # Phase 2 — full unfreeze, BN frozen
    print("\n" + "="*60)
    print("PHASE 2 - Full backbone, BN frozen, low LR")
    print("="*60)
    set_phase2_trainability(model, base)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=P2_LR),
        loss="categorical_crossentropy", metrics=["accuracy"],
    )
    h2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=P2_EPOCHS, callbacks=get_callbacks(2),
        class_weight=class_weights, verbose=1,
    )
    best_p2 = os.path.join(CKPT_DIR, "best_phase2.h5")
    if os.path.exists(best_p2):
        model.load_weights(best_p2)
        print(f"Loaded best phase-2 weights from {best_p2}")

    return h1, h2


def plot_history(h1, h2):
    acc   = h1.history["accuracy"]     + h2.history["accuracy"]
    val   = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss  = h1.history["loss"]         + h2.history["loss"]
    vloss = h1.history["val_loss"]     + h2.history["val_loss"]
    ep    = range(1, len(acc) + 1)
    p2    = len(h1.history["accuracy"]) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ep, acc, label="train acc")
    ax1.plot(ep, val, label="val acc")
    ax1.axvline(p2, color="gray", linestyle="--", label="Phase2 start")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.set_xlabel("Epoch")
    ax2.plot(ep, loss, label="train loss")
    ax2.plot(ep, vloss, label="val loss")
    ax2.axvline(p2, color="gray", linestyle="--")
    ax2.set_title("Loss"); ax2.legend(); ax2.set_xlabel("Epoch")
    plt.tight_layout()
    out = os.path.join(OUTPUT_ROOT, "training_curves.png")
    plt.savefig(out, dpi=120)
    print(f"Training curves saved to {out}")


def build_val_generator_raw(batch_size: int = BATCH_SIZE):
    """Raw [0..255] float32 — no rescale, matches MobileNetV3 internal Rescaling."""
    gen = ImageDataGenerator(validation_split=VAL_SPLIT)
    return gen.flow_from_directory(
        DATASET_PATH, target_size=IMG_SIZE, color_mode="rgb",
        classes=CLASSES, class_mode="categorical",
        batch_size=batch_size, subset="validation",
        seed=SEED, shuffle=False,
    )


def representative_dataset_gen(n_samples: int = 200):
    cal_ds = build_val_generator_raw(batch_size=1)
    count = 0
    cal_ds.reset()
    for images, _ in cal_ds:
        if count >= n_samples:
            return
        yield [np.array(images[0], dtype=np.float32)[np.newaxis]]
        count += 1


def quantize_to_int8(model, val_ds) -> str:
    print("\n" + "="*60)
    print("Dynamic Range Quantisation (best accuracy, any-quantization allowed)")
    print("="*60)
    saved_model_path = os.path.join(MODEL_DIR, "saved_model_float32")
    model.save(saved_model_path)
    print(f"SavedModel exported: {saved_model_path}")

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Dynamic range: weights→INT8, activations→float32 at runtime
    # Best accuracy without full-integer constraints

    tflite_model = converter.convert()
    out_path = os.path.join(TFLITE_DIR, "model_int8.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Dynamic range TFLite saved: {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)")
    return out_path


def evaluate_tflite(tflite_path: str, val_ds) -> float:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp  = interpreter.get_input_details()[0]
    outp = interpreter.get_output_details()[0]
    scale_in, zp_in = inp["quantization"]
    print(f"Input  dtype : {inp['dtype']}  shape: {inp['shape']}")
    print(f"Input  quant : scale={scale_in:.6f}, zp={zp_in}")

    total_samples = val_ds.samples
    seen = correct = 0
    val_ds.reset()
    for images, labels in val_ds:
        if seen >= total_samples:
            break
        for i in range(images.shape[0]):
            if seen >= total_samples:
                break
            img = np.array(images[i], dtype=np.float32)
            lbl = np.array(labels[i])
            if inp["dtype"] == np.int8:
                img_in = np.round(img / scale_in + zp_in).clip(-128, 127).astype(np.int8)
            else:
                img_in = img  # float32 dynamic model — pass raw [0..255]
            interpreter.set_tensor(inp["index"], img_in[np.newaxis])
            interpreter.invoke()
            out_q = interpreter.get_tensor(outp["index"])[0]
            correct += (int(np.argmax(out_q)) == int(np.argmax(lbl)))
            seen += 1

    acc = correct / total_samples if total_samples else 0.0
    print(f"TFLite accuracy on {total_samples} val samples: {acc*100:.2f}%")
    return acc


def load_model_for_quantize() -> keras.Model:
    for path in [
        os.path.join(CKPT_DIR, "best_p3_finetune.h5"),
        os.path.join(CKPT_DIR, "best_phase3.h5"),
        os.path.join(CKPT_DIR, "best_phase2.h5"),
        os.path.join(MODEL_DIR, "mobilenetv3_defect_float32.h5"),
    ]:
        if os.path.exists(path):
            print(f"Loading {path}")
            return keras.models.load_model(path, compile=False)
    raise FileNotFoundError("No trained model found in outputs/checkpoints or outputs/models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "quantize_eval", "eval_int8"],
        default="train",
    )
    parser.add_argument("--tflite", default=os.path.join(TFLITE_DIR, "model_int8.tflite"))
    args = parser.parse_args()

    print(f"TensorFlow {tf.__version__}  |  GPU: {tf.config.list_physical_devices('GPU')}")

    if args.mode == "eval_int8":
        val_raw = build_val_generator_raw(batch_size=BATCH_SIZE)
        evaluate_tflite(args.tflite, val_raw)
        return

    if args.mode == "quantize_eval":
        model = load_model_for_quantize()
        val_raw = build_val_generator_raw(batch_size=BATCH_SIZE)
        tflite_path = quantize_to_int8(model, val_raw)
        val_raw = build_val_generator_raw(batch_size=BATCH_SIZE)
        evaluate_tflite(tflite_path, val_raw)
        return

    # mode == "train"
    train_ds, val_ds = build_generators()
    print(f"Train batches: {len(train_ds)}  |  Val batches: {len(val_ds)}")
    print(f"Class indices: {train_ds.class_indices}")

    model, base = build_model()
    h1, h2 = train(model, base, train_ds, val_ds)
    plot_history(h1, h2)

    final_h5 = os.path.join(MODEL_DIR, "mobilenetv3_defect_float32.h5")
    model.save(final_h5)
    print(f"Float32 model saved: {final_h5}")

    val_ds.reset()
    results = model.evaluate(val_ds, verbose=1)
    print(f"Float32 val  loss={results[0]:.4f}  acc={results[1]*100:.2f}%")

    val_raw = build_val_generator_raw(batch_size=BATCH_SIZE)
    tflite_path = quantize_to_int8(model, val_raw)
    val_raw = build_val_generator_raw(batch_size=BATCH_SIZE)
    evaluate_tflite(tflite_path, val_raw)


if __name__ == "__main__":
    main()