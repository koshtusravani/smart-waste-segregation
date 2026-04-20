import os
import shutil
import random
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import (
    DATA_RAW_DIR, DATA_TRAIN_DIR, DATA_VAL_DIR, DATA_TEST_DIR,
    TRASHNET_REMAP, CLASS_NAMES,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED
)

random.seed(RANDOM_SEED)



def download_trashnet():
    
    import zipfile
    import urllib.request

    zip_url  = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
    zip_path = os.path.join(DATA_RAW_DIR, "dataset-resized.zip")
    raw_out  = os.path.join(DATA_RAW_DIR, "trashnet_images")

    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    if not os.path.exists(zip_path):
        print(f"[Setup] Downloading TrashNet (~44MB)...")
        print(f"        From: {zip_url}")

        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            pct = min(downloaded / total_size * 100, 100)
            bar = int(pct / 2)
            print(f"\r        [{'=' * bar}{' ' * (50 - bar)}] {pct:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(zip_url, zip_path, progress)
        print("\n[Setup] Download complete.")
    else:
        print(f"[Setup] Zip already exists, skipping download.")

    if not os.path.exists(raw_out):
        print(f"[Setup] Extracting zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_RAW_DIR)

        extracted = os.path.join(DATA_RAW_DIR, "dataset-resized")
        if os.path.exists(extracted):
            os.rename(extracted, raw_out)

        print(f"[Setup] Extracted to {raw_out}")
    else:
        print(f"[Setup] Already extracted, skipping.")

    print("\n[Setup] Found classes:")
    for cls in sorted(os.listdir(raw_out)):
        cls_path = os.path.join(raw_out, cls)
        if os.path.isdir(cls_path):
            count = len(os.listdir(cls_path))
            print(f"   {cls:15s}: {count} images")

    return raw_out


def remap_classes(raw_dir):
    print("[Setup] Remapping classes to 5-class system...")
    class_images = {cls: [] for cls in CLASS_NAMES}

    for original_class, target_class in TRASHNET_REMAP.items():
        src_dir = os.path.join(raw_dir, original_class)
        if not os.path.isdir(src_dir):
            print(f"   [WARN] Directory not found: {src_dir} — skipping.")
            continue
        images = [os.path.join(src_dir, f) for f in os.listdir(src_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        class_images[target_class].extend(images)
        print(f"   {original_class:12s} → {target_class:10s}: {len(images)} images")

    print("\n[Setup] Final class sizes:")
    for cls, imgs in class_images.items():
        print(f"   {cls:12s}: {len(imgs)} images")

    return class_images

def split_and_copy(class_images):
    print(f"\n[Setup] Splitting: {TRAIN_SPLIT:.0%} train | "
          f"{VAL_SPLIT:.0%} val | {TEST_SPLIT:.0%} test")

    for split_dir in [DATA_TRAIN_DIR, DATA_VAL_DIR, DATA_TEST_DIR]:
        os.makedirs(split_dir, exist_ok=True)

    for cls, images in class_images.items():
        random.shuffle(images)
        n = len(images)
        n_train = int(n * TRAIN_SPLIT)
        n_val   = int(n * VAL_SPLIT)

        splits = {
            DATA_TRAIN_DIR: images[:n_train],
            DATA_VAL_DIR:   images[n_train:n_train + n_val],
            DATA_TEST_DIR:  images[n_train + n_val:],
        }

        for split_dir, split_images in splits.items():
            cls_out = os.path.join(split_dir, cls)
            os.makedirs(cls_out, exist_ok=True)
            for img_path in split_images:
                dest = os.path.join(cls_out, os.path.basename(img_path))
                
                if os.path.exists(dest):
                    base, ext = os.path.splitext(os.path.basename(img_path))
                    dest = os.path.join(cls_out, f"{base}_{random.randint(0,9999)}{ext}")
                shutil.copy2(img_path, dest)

        split_dir_counts = {
            "train": len(splits[DATA_TRAIN_DIR]),
            "val":   len(splits[DATA_VAL_DIR]),
            "test":  len(splits[DATA_TEST_DIR]),
        }
        print(f"   {cls:12s}: train={split_dir_counts['train']} | "
              f"val={split_dir_counts['val']} | test={split_dir_counts['test']}")

    print("\n[Setup]  Dataset ready in data/processed/")

def main():
    
    if os.path.exists(DATA_TRAIN_DIR) and os.listdir(DATA_TRAIN_DIR):
        print("[Setup] Processed data already exists. Delete data/processed/ to re-run.")
        return

    raw_dir     = download_trashnet()
    class_imgs  = remap_classes(raw_dir)
    split_and_copy(class_imgs)


if __name__ == "__main__":
    main()