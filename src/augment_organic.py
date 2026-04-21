# augment_organic.py
# Hema Sravani Koshtu
# April 20, 2026
# purpose: offline augmentation to balance underrepresented classes before training

import os
import sys
import random
from PIL import Image, ImageEnhance, ImageFilter
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from config import DATA_TRAIN_DIR, RANDOM_SEED

random.seed(RANDOM_SEED)

#target image count per class after augmentation
AUGMENT_TARGETS = {
    "organic": 900,
    "metal":   500,
    "glass":   900,
    "plastic": 500,
}


def augment_image(img):
    #randomly selects 2-4 augmentation operations and applies them to a single image
    ops = random.sample([
        lambda i: i.transpose(Image.FLIP_LEFT_RIGHT),
        lambda i: i.transpose(Image.FLIP_TOP_BOTTOM),
        lambda i: i.rotate(random.randint(-30, 30), expand=False, fillcolor=(128,128,128)),
        lambda i: ImageEnhance.Brightness(i).enhance(random.uniform(0.6, 1.4)),
        lambda i: ImageEnhance.Contrast(i).enhance(random.uniform(0.6, 1.4)),
        lambda i: ImageEnhance.Color(i).enhance(random.uniform(0.6, 1.5)),
        lambda i: ImageEnhance.Sharpness(i).enhance(random.uniform(0.5, 2.0)),
        lambda i: i.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5))),
        lambda i: i.crop((
            random.randint(0, 30), random.randint(0, 30),
            i.width - random.randint(0, 30), i.height - random.randint(0, 30)
        )).resize(i.size),
    ], k=random.randint(2, 4))

    for op in ops:
        img = op(img)
    return img


def augment_class(cls_name, target_count):
    cls_dir = os.path.join(DATA_TRAIN_DIR, cls_name)
    if not os.path.isdir(cls_dir):
        print(f"[Augment] Skipping {cls_name} — directory not found.")
        return

    existing = [f for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    current_count = len(existing)

    #skip if the class already meets the target count
    if current_count >= target_count:
        print(f"[Augment] {cls_name}: already has {current_count} images, skipping.")
        return

    needed = target_count - current_count
    print(f"[Augment] {cls_name}: {current_count} → {target_count} (generating {needed} images)...")

    #generate augmented images and save them to the same class directory
    generated = 0
    while generated < needed:
        src_file = random.choice(existing)
        src_path = os.path.join(cls_dir, src_file)
        try:
            img = Image.open(src_path).convert("RGB")
            aug = augment_image(img)
            base, ext = os.path.splitext(src_file)
            out_name = f"aug_{generated:05d}_{base}{ext}"
            aug.save(os.path.join(cls_dir, out_name), quality=90)
            generated += 1
        except Exception as e:
            print(f"   [WARN] Failed on {src_file}: {e}")

    print(f"[Augment] {cls_name}: done. New count: {len(os.listdir(cls_dir))} images.")


def main():
    print("[Augment] Starting class balancing via image augmentation...\n")
    for cls_name, target in AUGMENT_TARGETS.items():
        augment_class(cls_name, target)
    print("\n[Augment] All done. Re-run python src/train.py to retrain.")


if __name__ == "__main__":
    main()