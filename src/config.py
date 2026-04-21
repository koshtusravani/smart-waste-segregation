# config.py
# Hema Sravani Koshtu
# April 20, 2026
# purpose: central configuration file for all paths, hyperparameters, and class mappings

import os

#paths
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
DATA_TRAIN_DIR  = os.path.join(BASE_DIR, "data", "processed", "train")
DATA_VAL_DIR    = os.path.join(BASE_DIR, "data", "processed", "val")
DATA_TEST_DIR   = os.path.join(BASE_DIR, "data", "processed", "test")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
PLOTS_DIR       = os.path.join(RESULTS_DIR, "plots")
CM_DIR          = os.path.join(RESULTS_DIR, "confusion_matrix")

#best model checkpoint saved during training
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

#classes
CLASS_NAMES = ["glass", "metal", "organic", "paper", "plastic"]

#maps trashnet's 6 original classes to the 5-class system
TRASHNET_REMAP = {
    "cardboard": "paper",
    "glass":     "glass",
    "metal":     "metal",
    "paper":     "paper",
    "plastic":   "plastic",
    "trash":     "organic",
}

NUM_CLASSES = len(CLASS_NAMES)

#disposal recommendations displayed in the real-time interface
DISPOSAL_RECOMMENDATIONS = {
    "glass":    "♻️  Dispose in the Glass Recycling Bin",
    "metal":    "♻️  Dispose in the Metal / Recycling Bin",
    "organic":  "🌱  Dispose in the Compost / Green Waste Bin",
    "paper":    "📄  Dispose in the Paper Recycling Bin",
    "plastic":  "🧴  Dispose in the Plastic Recycling Bin",
}

#image settings
IMG_SIZE   = 224
MEAN       = [0.485, 0.456, 0.406]   #imagenet mean
STD        = [0.229, 0.224, 0.225]   #imagenet std

#dataset split ratio
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
RANDOM_SEED = 42

#training hyperparameters
BATCH_SIZE    = 32
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
LR_STEP_SIZE  = 10
LR_GAMMA      = 0.1

#real-time inference setings
CONFIDENCE_THRESHOLD   = 0.50
HIGH_CONF_COLOR        = (0, 200, 0)     #green for high confidence
LOW_CONF_COLOR         = (0, 0, 220)     #red for low confidence
WARNING_COLOR          = (0, 165, 255)   #orange when below threshold
CAMERA_INDEX           = 0              #default webcam

#set to true to enable voice announcements via pyttsx3
ENABLE_VOICE = True