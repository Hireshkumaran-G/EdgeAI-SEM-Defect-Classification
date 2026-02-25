"""
Augmentation script — run BEFORE train_mcu.py
Expands each class to TARGET_COUNT images using light augmentation.
"""
import os
import random
import argparse
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img

SEED = 42  # Set random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR     = os.getcwd()
SOURCE_DIR   = os.path.join(BASE_DIR, "Hackathon_phase3_training_dataset")
OUTPUT_DIR   = os.path.join(BASE_DIR, "Hackathon_phase3_training_dataset_augmented")
TARGET_COUNT = 400
IMG_SIZE     = (128, 128)
CLASSES = [
    "BRIDGE","CLEAN_CRACK","CLEAN_LAYER","CLEAN_VIA",
    "CMP","CRACK","LER","OPEN","OTHERS","PARTICLE","VIA"
]

auggen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.08,
    zoom_range=0.12,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest",
)  # Augmentation generator

def load_grayscale(path: str) -> np.ndarray:
    """Load image as grayscale uint8 (H,W,1)."""
    img = Image.open(path).convert("L").resize(IMG_SIZE)
    return img_to_array(img)          # shape (128,128,1), float32 0-255

def save_image(arr: np.ndarray, path: str):
    img = array_to_img(arr)  
    img.save(path)  

def augment_class(class_name: str, src_dir: str, dst_dir: str, target: int):
    os.makedirs(dst_dir, exist_ok=True)  

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    src_files = [
        f for f in os.listdir(src_dir)
        if os.path.splitext(f)[1].lower() in exts
    ] 
    if not src_files:
        print(f"  [WARN] No images found in {src_dir}, skipping.")
        return 0

    for fname in src_files:
        dst = os.path.join(dst_dir, fname)
        if not os.path.exists(dst):
            Image.open(os.path.join(src_dir, fname)).convert("L").resize(IMG_SIZE).save(dst) 

    existing = len(os.listdir(dst_dir))  
    needed   = max(0, target - existing)  
    if needed == 0:
        print(f"  {class_name:15s}: {existing} images (no augmentation needed)")
        return existing

    generated = 0
    rng = np.random.RandomState(SEED + hash(class_name) % 10000)  
    while generated < needed:
        fname = src_files[rng.randint(0, len(src_files))]  
        arr   = load_grayscale(os.path.join(src_dir, fname)).reshape(1, *IMG_SIZE, 1)  
        for batch in auggen.flow(arr, batch_size=1, seed=int(rng.randint(0, 99999))):
            save_image(batch[0], os.path.join(dst_dir, f"aug_{class_name}_{generated:05d}.png"))  
            generated += 1
            break 

    final = len(os.listdir(dst_dir))  
    print(f"  {class_name:15s}: {len(src_files):3d} → {final} (+{generated} augmented)")
    return final

def main(args):
    print(f"Source : {args.source}\nOutput : {args.output}\nTarget : {args.target}/class\n")
    total, missing = 0, []
    for cls in CLASSES:
        src = os.path.join(args.source, cls)  
        dst = os.path.join(args.output, cls)  
        if not os.path.isdir(src):
            print(f"  [WARN] Missing: {src}")
            missing.append(cls)
            continue
        total += augment_class(cls, src, dst, args.target)  
    print(f"\nDone. Total images: {total}")
    if missing:
        print(f"Missing classes: {missing}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument("--source", default=SOURCE_DIR)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--target", type=int, default=TARGET_COUNT)
    main(parser.parse_args()) 