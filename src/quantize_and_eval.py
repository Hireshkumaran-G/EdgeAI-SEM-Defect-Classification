"""
Standalone quantization & evaluation script.
Converts saved_model_float32 to multiple TFLite variants and evaluates each.
"""
import os, pathlib, numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR        = os.getcwd()  # base directory for reproducibility
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models", "saved_model_float32")
TFLITE_OUT_DIR  = os.path.join(BASE_DIR, "outputs", "tflite")
DATASET_DIR     = os.path.join(BASE_DIR, "Hackathon_phase3_training_dataset_augmented")
IMG_SIZE        = (128, 128)
BATCH_SIZE      = 32
VAL_SPLIT       = 0.2
SEED            = 42
CLASSES = [
    "BRIDGE","CLEAN_CRACK","CLEAN_LAYER","CLEAN_VIA",
    "CMP","CRACK","LER","OPEN","OTHERS","PARTICLE","VIA"
]

os.makedirs(TFLITE_OUT_DIR, exist_ok=True)  # ensure output directory exists


def make_raw_generator(subset):
    """Raw [0..255] float32 — no rescale, matches MobileNetV3 internal Rescaling."""
    gen = ImageDataGenerator(validation_split=VAL_SPLIT)
    return gen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, color_mode="rgb",
        classes=CLASSES, class_mode="categorical",
        batch_size=BATCH_SIZE, subset=subset, seed=SEED,
        shuffle=(subset == "training"),
    )


def make_representative_dataset(n_samples=200, subset="validation"):
    cal_ds = ImageDataGenerator(validation_split=VAL_SPLIT).flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, color_mode="rgb",
        classes=CLASSES, class_mode="categorical",
        batch_size=1, subset=subset, seed=SEED, shuffle=False,
    )
    def representative_dataset():
        cal_ds.reset()
        count = 0
        for images, _ in cal_ds:
            if count >= n_samples:
                return
            yield [np.array(images[0], dtype=np.float32)[np.newaxis]]
            count += 1
    return representative_dataset


def convert_and_save(converter, out_path):
    tflite_model = converter.convert()
    pathlib.Path(out_path).write_bytes(tflite_model)
    print(f"Saved {out_path}  ({len(tflite_model)/1024:.1f} KB)")
    return out_path


def evaluate_tflite(tflite_path):
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    scale_in, zp_in = inp["quantization"]
    print(f"  input  dtype={inp['dtype']}  shape={inp['shape']}")
    print(f"  Input quant: scale={scale_in:.6f}, zp={zp_in}")

    flow = make_raw_generator("validation")
    correct = total = 0
    flow.reset()
    for i, (imgs, labels) in enumerate(flow):
        for img, label in zip(imgs, labels):
            x = img[np.newaxis].astype(np.float32)
            if inp["dtype"] == np.int8:
                x = np.round(x / scale_in + zp_in).clip(-128, 127).astype(np.int8)
            elif inp["dtype"] == np.uint8:
                x = np.round(x / scale_in + zp_in).clip(0, 255).astype(np.uint8)
            # float32: pass raw [0..255] as-is
            interp.set_tensor(inp["index"], x)
            interp.invoke()
            pred = np.argmax(interp.get_tensor(out["index"]))
            correct += int(pred == np.argmax(label))
            total   += 1
        if (i + 1) >= len(flow):
            break
    print(f"  Accuracy: {correct}/{total} = {100*correct/total:.2f}%")
    return correct / total


# ── 1. Dynamic-range (PRIMARY submission — best accuracy) ────────────────────
print("\n--- Dynamic-range (PRIMARY) ---")
conv = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
convert_and_save(conv, os.path.join(TFLITE_OUT_DIR, "model_dynamic.tflite"))
evaluate_tflite(os.path.join(TFLITE_OUT_DIR, "model_dynamic.tflite"))

# ── 2. Full-INT8 with float fallback (backup) ────────────────────────────────
print("\n--- Full-INT8 with float fallback (BACKUP) ---")
conv = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = make_representative_dataset(n_samples=200)
conv.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,
]
conv.inference_input_type  = tf.int8
conv.inference_output_type = tf.int8
convert_and_save(conv, os.path.join(TFLITE_OUT_DIR, "model_int8_fallback.tflite"))
evaluate_tflite(os.path.join(TFLITE_OUT_DIR, "model_int8_fallback.tflite"))

# ── 3. Strict full-integer INT8 (MCU-only backup) ────────────────────────────
print("\n--- Strict full-integer INT8 (MCU backup) ---")
conv = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = make_representative_dataset(n_samples=200)
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type  = tf.int8
conv.inference_output_type = tf.int8
convert_and_save(conv, os.path.join(TFLITE_OUT_DIR, "model_int8_strict.tflite"))
evaluate_tflite(os.path.join(TFLITE_OUT_DIR, "model_int8_strict.tflite"))
