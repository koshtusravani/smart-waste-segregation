# Smart Waste Segregation Assistant
A Real-Time Eco-Friendly AI System for Multi-Class Waste Classification
Final Project — Hema Sravani Koshtu

## Project Overview
I built this system to address a problem I noticed — most people do not know how to correctly sort their waste, which leads to recyclable materials ending up in landfills. The Smart Waste Segregation Assistant uses computer vision and deep learning to classify waste in real time through a webcam into five categories: glass, metal, organic, paper, and plastic. Once classified, the system displays the predicted category, confidence level, and a disposal recommendation directly on screen.

The model is built on MobileNetV2, a lightweight CNN pretrained on ImageNet, and fine-tuned on a combined dataset of TrashNet and RealWaste. Training used a two-phase approach — first training only the classification head with the backbone frozen, then unfreezing all layers for full fine-tuning. The final model achieves 90.97% accuracy on the test set across all five classes.

## Project Structure
smart_waste_segregation/
├── data/
│   ├── raw/                         #downloaded source datasets
│   └── processed/
│       ├── train/                   #70% split
│       ├── val/                     #15% split
│       └── test/                    #15% split
├── src/
│   ├── config.py                    #all hyperparameters and paths
│   ├── dataset.py                   #dataset loading and transforms
│   ├── model.py                     #mobileNetV2 architecture
│   ├── train.py                     #5-class training script
│   ├── train_4class.py              #4-class training (glass excluded)
│   ├── evaluate.py                  #5-class evaluation
│   ├── evaluate_4class.py           #4-class evaluation
│   ├── compare_models.py            #4-class vs 5-class comparison
│   └── realtime.py                  #webcam real-time inference
├── models/                          #saved .pth checkpoints
├── results/
│   ├── plots/                       #loss and accuracy curves (5-class)
│   ├── plots_4class/                #loss and accuracy curves (4-class)
│   ├── confusion_matrix/            #confusion matrix (5-class)
│   ├── confusion_matrix_4class/     #confusion matrix (4-class)
│   └── comparison/                  #F1 bar chart comparing both models
├── tests/
│   └── test_model.py                #unit tests
├── setup_combined_dataset.py        #download + combine + split dataset
├── augment_organic.py               #balance underrepresented classes
└── requirements.txt

## Quick Start

### 1. Install dependencies
pip install -r requirements.txt

### 2. Prepare the dataset
python setup_combined_dataset.py

This downloads TrashNet, expects RealWaste to be pre-downloaded at data/raw/realwaste-main/, merges both datasets, remaps to 5 classes, and splits into train/val/test.

### 3. Balance classes with augmentation
python augment_organic.py

Generates augmented images for underrepresented classes (glass, organic) to bring them up to 900 images each.

### 4. Train the 5-class model
python src/train.py

### 5. Evaluate the 5-class model
python src/evaluate.py

### 6. Train and evaluate the 4-class model (optional comparison)
python src/train_4class.py
python src/evaluate_4class.py
python src/compare_models.py

### 7. Run the real-time webcam demo
python src/realtime.py

Press s to save a screenshot. Press q to quit.

## Results

### 5-Class Model

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| glass | 0.9328 | 0.8993 | 0.9158 |
| metal | 0.7991 | 0.9722 | 0.8772 |
| organic | 0.9048 | 0.9157 | 0.9102 |
| paper | 0.9658 | 0.9559 | 0.9608 |
| plastic | 0.9385 | 0.7962 | 0.8615 |
| macro avg | 0.9082 | 0.9079 | 0.9051 |

Overall Test Accuracy: 90.97%

### 4-Class vs 5-Class Comparison

| Class | 4-Class F1 | 5-Class F1 | Change |
|-------|-----------|-----------|--------|
| metal | 0.9081 | 0.8772 | +0.031 |
| organic | 0.9333 | 0.9102 | +0.023 |
| paper | 0.9641 | 0.9608 | ~same |
| plastic | 0.8894 | 0.8615 | +0.028 |

The 4-class model (glass excluded) achieves 92.72% accuracy compared to 90.97% for the 5-class model. The improvement in plastic and organic F1 when glass is removed confirms the visual similarity between glass and plastic noted in the proposal. Despite this, the 5-class model is the correct design choice since it provides specific disposal guidance for glass, which the 4-class model cannot do.

## Dataset
The system was trained on a combination of two datasets:

- TrashNet — 2,527 images across 6 classes, remapped to 5 (cardboard to paper, trash to organic)
- RealWaste — real-world waste images, selected classes mapped to the same 5-category system

Final training set: 4,992 images across 5 classes after augmentation.

## Tech Stack
- Python 3.10+
- PyTorch — model training and inference
- torchvision — MobileNetV2, image transforms
- OpenCV — webcam capture and real-time overlay
- scikit-learn — evaluation metrics
- matplotlib / seaborn — plots and confusion matrices
- Pillow — image augmentation

## Author
Hema Sravani Koshtu
Final Project — Individual Submission