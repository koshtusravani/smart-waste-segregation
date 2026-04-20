import os, sys, shutil, random, zipfile
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from config import (
    DATA_RAW_DIR, DATA_TRAIN_DIR, DATA_VAL_DIR, DATA_TEST_DIR,
    CLASS_NAMES, TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED
)

random.seed(RANDOM_SEED)

TRASHNET_REMAP = {
    "cardboard": "paper",
    "glass":     "glass",
    "metal":     "metal",
    "paper":     "paper",
    "plastic":   "plastic",
    "trash":     "organic",
}

REALWASTE_REMAP = {
    "Cardboard":          "paper",
    "Food Organics":      "organic",
    "Glass":              "glass",
    "Metal":              "metal",
    "Paper":              "paper",
    "Plastic":            "plastic",
    "Miscellaneous Trash": None,   
    "Textile Trash":       None,   
    "Vegetation":          None,   
}


def progress_bar(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100)
    bar = int(pct / 2)
    print(f"\r   [{'=' * bar}{' ' * (50-bar)}] {pct:.1f}%", end="", flush=True)


def download_trashnet():
    zip_url  = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
    zip_path = os.path.join(DATA_RAW_DIR, "trashnet.zip")
    out_dir  = os.path.join(DATA_RAW_DIR, "trashnet_images")
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    if not os.path.exists(zip_path):
        print("[Setup] Downloading TrashNet (~44MB)...")
        urllib.request.urlretrieve(zip_url, zip_path, progress_bar)
        print()

    if not os.path.exists(out_dir):
        print("[Setup] Extracting TrashNet...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_RAW_DIR)
        extracted = os.path.join(DATA_RAW_DIR, "dataset-resized")
        if os.path.exists(extracted):
            os.rename(extracted, out_dir)

    print(f"[Setup] TrashNet ready at {out_dir}")
    return out_dir

def download_realwaste():
    out_dir = os.path.join(DATA_RAW_DIR, "realwaste-main", "RealWaste")
    print(f"[Setup] RealWaste found at {out_dir}")
    return out_dir


def collect_images(raw_dir, remap):
    class_images = {cls: [] for cls in CLASS_NAMES}
    for original, target in remap.items():
        if target is None:
            continue
        candidates = [
            os.path.join(raw_dir, original),
        ]
        src_dir = next((c for c in candidates if os.path.isdir(c)), None)
        if not src_dir:
            print(f"   [WARN] Not found: {original} in {raw_dir}")
            continue
        imgs = [os.path.join(src_dir, f) for f in os.listdir(src_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        class_images[target].extend(imgs)
        print(f"   {original:22s} → {target:10s}: {len(imgs)} images")
    return class_images


def merge(a, b):
    merged = {cls: a.get(cls, []) + b.get(cls, []) for cls in CLASS_NAMES}
    return merged


def split_and_copy(class_images):
    print(f"\n[Setup] Final class sizes:")
    for cls, imgs in class_images.items():
        print(f"   {cls:12s}: {len(imgs)} images")

    print(f"\n[Setup] Splitting 70% train | 15% val | 15% test...")
    for d in [DATA_TRAIN_DIR, DATA_VAL_DIR, DATA_TEST_DIR]:
        os.makedirs(d, exist_ok=True)

    for cls, images in class_images.items():
        random.shuffle(images)
        n       = len(images)
        n_train = int(n * TRAIN_SPLIT)
        n_val   = int(n * VAL_SPLIT)
        splits  = {
            DATA_TRAIN_DIR: images[:n_train],
            DATA_VAL_DIR:   images[n_train:n_train+n_val],
            DATA_TEST_DIR:  images[n_train+n_val:],
        }
        for split_dir, split_imgs in splits.items():
            cls_out = os.path.join(split_dir, cls)
            os.makedirs(cls_out, exist_ok=True)
            for i, img_path in enumerate(split_imgs):
                ext  = os.path.splitext(img_path)[1]
                dest = os.path.join(cls_out, f"{cls}_{i:05d}{ext}")
                shutil.copy2(img_path, dest)
        print(f"   {cls:12s}: train={len(splits[DATA_TRAIN_DIR])} | "
              f"val={len(splits[DATA_VAL_DIR])} | test={len(splits[DATA_TEST_DIR])}")

    print("\n[Setup]  Combined dataset ready in data/processed/")


def main():
    if os.path.exists(DATA_TRAIN_DIR) and os.listdir(DATA_TRAIN_DIR):
        print("[Setup] data/processed/ already exists.")
        print("        Delete it to rebuild: rmdir /s /q data\\processed")
        return

    print("=" * 60)
    print(" Combining TrashNet + RealWaste → 5-class dataset")
    print("=" * 60)

    trashnet_dir  = download_trashnet()
    realwaste_dir = download_realwaste()

    print("\n[Setup] Collecting TrashNet images...")
    trashnet_imgs = collect_images(trashnet_dir, TRASHNET_REMAP)

    print("\n[Setup] Collecting RealWaste images...")
    realwaste_imgs = collect_images(realwaste_dir, REALWASTE_REMAP)

    print("\n[Setup] Merging both datasets...")
    combined = merge(trashnet_imgs, realwaste_imgs)

    split_and_copy(combined)


if __name__ == "__main__":
    main()