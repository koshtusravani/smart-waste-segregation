# ♻️ Smart Waste Segregation Assistant

A Real-Time Eco-Friendly AI System for Multi-Class Waste Classification
> PRCV Final Project — Hema Sravani Koshtu

---

## 📌 Project Overview

This system uses computer vision + deep learning to classify waste in real-time via webcam into 5 categories:

| Class | Examples |
|-------|----------|
| 🧴 Plastic | Bottles, bags, containers |
| 📄 Paper | Cardboard, newspapers, boxes |
| 🥫 Metal | Cans, foil, tins |
| 🍌 Organic | Food scraps, leaves, general waste |
| 🪟 Glass | Bottles, jars, broken glass |

Real-time predictions include category, confidence level, and disposal recommendation.

---

## 🗂️ Project Structure

smart_waste_segregation/
├── data/
│   ├── raw/                    # Original dataset (TrashNet or custom)
│   └── processed/
│       ├── train/              # 70% split
│       ├── val/                # 15% split
│       └── test/               # 15% split
├── src/
│   ├── config.py               # All hyperparameters and paths
│   ├── dataset.py              # Dataset loading and augmentation
│   ├── model.py                # CNN architecture
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Metrics: accuracy, precision, recall, F1
│   └── realtime.py             # Webcam real-time inference
├── models/                     # Saved .pth model checkpoints
├── notebooks/
│   └── exploration.ipynb       # EDA and training experiments
├── results/
│   ├── plots/                  # Training/validation loss & accuracy curves
│   └── confusion_matrix/       # Confusion matrix images
├── tests/
│   └── test_model.py           # Unit tests
├── requirements.txt
├── setup_dataset.py            # Download + split dataset automatically
└── README.md


---

## 🚀 Quick Start

### 1. Install Dependencies
pip install -r requirements.txt

### 2. Download & Prepare Dataset
python setup_dataset.py
> This downloads TrashNet, remaps to 5 classes, and splits into train/val/test.

### 3. Train the Model
python src/train.py

### 4. Evaluate
python src/evaluate.py

### 5. Run Real-Time Webcam Demo
python src/realtime.py

---

## 📊 Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score (per class + macro avg)
- Confusion Matrix (especially glass/plastic and paper/organic)
- Real-time webcam performance on unseen objects

---

## 🔧 Tech Stack
- Python 3.10+
- PyTorch — CNN model training
- OpenCV — webcam capture and real-time display
- torchvision — image transforms and augmentations
- scikit-learn — evaluation metrics
- matplotlib / seaborn — plots

---

## 📁 Dataset

Recommended: TrashNet
- Download: https://huggingface.co/datasets/garythung/trashnet
- 6 original classes → remapped to 5 (cardboard → paper, trash → organic)
- ~2,500 images total

---

## 👩‍💻 Author
Hema Sravani Koshtu