"""
config.py
---------
Central configuration for the Smart Waste Segregation project.
All paths, hyperparameters, and class mappings live here.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
DATA_TRAIN_DIR  = os.path.join(BASE_DIR, "data", "processed", "train")
DATA_VAL_DIR    = os.path.join(BASE_DIR, "data", "processed", "val")
DATA_TEST_DIR   = os.path.join(BASE_DIR, "data", "processed", "test")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
PLOTS_DIR       = os.path.join(RESULTS_DIR, "plots")
CM_DIR          = os.path.join(RESULTS_DIR, "confusion_matrix")

# Best model checkpoint path
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# ── Classes ───────────────────────────────────────────────────────────────────
# 5-class system as per project proposal
CLASS_NAMES = ["glass", "metal", "organic", "paper", "plastic"]

# Mapping from TrashNet's 6 original classes → our 5 classes
# TrashNet classes: cardboard, glass, metal, paper, plastic, trash
TRASHNET_REMAP = {
    "cardboard": "paper",    # cardboard → paper (as stated in proposal)
    "glass":     "glass",
    "metal":     "metal",
    "paper":     "paper",
    "plastic":   "plastic",
    "trash":     "organic",  # general waste → organic (as stated in proposal)
}

NUM_CLASSES = len(CLASS_NAMES)

# ── Disposal Recommendations ──────────────────────────────────────────────────
DISPOSAL_RECOMMENDATIONS = {
    "glass":    "♻️  Dispose in the Glass Recycling Bin",
    "metal":    "♻️  Dispose in the Metal / Recycling Bin",
    "organic":  "🌱  Dispose in the Compost / Green Waste Bin",
    "paper":    "📄  Dispose in the Paper Recycling Bin",
    "plastic":  "🧴  Dispose in the Plastic Recycling Bin",
}

# ── Image Settings ────────────────────────────────────────────────────────────
IMG_SIZE   = 224          # Resize all images to 224×224
MEAN       = [0.485, 0.456, 0.406]   # ImageNet mean (we use pretrained backbone)
STD        = [0.229, 0.224, 0.225]   # ImageNet std

# ── Dataset Split ─────────────────────────────────────────────────────────────
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
RANDOM_SEED = 42

# ── Training Hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
LR_STEP_SIZE  = 10        # StepLR: decay every N epochs
LR_GAMMA      = 0.1       # StepLR: multiply LR by this factor

# ── Real-Time Settings ────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD   = 0.50   # Below this → show warning
HIGH_CONF_COLOR        = (0, 200, 0)     # Green (BGR for OpenCV)
LOW_CONF_COLOR         = (0, 0, 220)     # Red (BGR for OpenCV)
WARNING_COLOR          = (0, 165, 255)   # Orange (BGR for OpenCV)
CAMERA_INDEX           = 0              # Default webcam

# ── Voice Output ──────────────────────────────────────────────────────────────
ENABLE_VOICE = False     # Set True to enable pyttsx3 voice announcements