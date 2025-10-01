# ML Model - Machine Learning Anemia Detection

## Overview

This module contains a lightweight RGB-based machine learning model for anemia detection from eye images. It's optimized for low-resource devices like mobile phones and embedded systems.

## Features

✅ **Multiple ML Algorithms**: Random Forest, Gradient Boosting, Logistic Regression, SVM  
✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, Specificity, AUC-ROC, Confidence Scores  
✅ **Fast Inference**: < 300ms on mobile devices  
✅ **Low Memory Footprint**: < 30 MB  
✅ **Color-Based Features**: RGB/HSV/LAB statistics, histograms, paleness index  
✅ **Dataset Integration**: Automatic download from Kaggle (Eyes Defy Anemia)

## Project Structure

```
ml_model/
├── data/
│   ├── load_dataset.py       # Download and load Eyes Defy Anemia dataset
│   ├── preprocess.py          # Image preprocessing and augmentation
│   ├── raw/                   # Raw downloaded data
│   ├── processed/             # Preprocessed train/val/test splits
│   └── features/              # Extracted features
│
├── features/
│   └── extract_features.py    # Color and morphological feature extraction
│
├── models/
│   └── (trained models saved here)
│
├── training/
│   └── train_ml_model.py     # Model training pipeline
│
├── inference/
│   └── predict.py             # Lightweight inference script
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### 1. Prerequisites

- Python 3.8+
- pip
- Kaggle account (for dataset download)

### 2. Set up Kaggle API

```bash
# Install kaggle
pip install kaggle

# Set up credentials (Windows)
# Copy kaggle.json to C:\Users\<YourUsername>\.kaggle\
```

### 3. Install Dependencies

```bash
cd ml_model
pip install -r requirements.txt
```

## Quick Start

### Step 1: Download Dataset

```bash
cd data
python load_dataset.py
```

This will download the Eyes Defy Anemia dataset from Kaggle and organize images.

### Step 2: Preprocess Data

```bash
python preprocess.py
```

Splits data into train/val/test, applies augmentation and preprocessing.

### Step 3: Extract Features

```bash
cd ..\features
python extract_features.py
```

Extracts RGB/HSV/LAB color features, paleness index, and texture features.

### Step 4: Train Models

```bash
cd ..\training
python train_ml_model.py
```

Trains multiple ML models and evaluates with comprehensive metrics.

### Step 5: Run Inference

```bash
cd ..\inference

# Single model prediction
python predict.py --image path\to\eye_image.jpg --model random_forest

# Compare all models
python predict.py --image path\to\eye_image.jpg --compare

# Save result to JSON
python predict.py --image path\to\eye_image.jpg --save result.json
```

## Performance Metrics

The model outputs comprehensive metrics including:

### Classification Metrics

- **Accuracy**: Overall prediction accuracy
- **Precision**: Positive predictive value
- **Recall/Sensitivity**: True positive rate
- **Specificity**: True negative rate
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

### Confidence Metrics

- **Confidence Score**: Model confidence for each prediction (0-1)
- **Class Probabilities**: Probability distribution across classes

### Execution Metrics

- **Preprocessing Time**: Image loading and preprocessing (ms)
- **Feature Extraction Time**: Time to extract features (ms)
- **Inference Time**: Model prediction time (ms)
- **Total Time**: End-to-end execution time (ms)

## Example Output

```json
{
  "prediction": {
    "class": "Anemia",
    "confidence_score": 0.87,
    "class_probabilities": {
      "Normal": 0.13,
      "Anemia": 0.87
    }
  },
  "execution_info": {
    "model_type": "random_forest",
    "os": "Windows",
    "inference_time_ms": 12.34,
    "total_time_ms": 136.13
  }
}
```

## Dataset

**Eyes Defy Anemia Dataset** from Kaggle:

- Dataset ID: `harshwardhanfartale/eyes-defy-anemia`
- Contains eye images labeled for anemia detection
- Automatically downloaded via `load_dataset.py`

## Notes

- Optimized for low-power devices and fast inference
- Models saved in `models/` directory
- Features extracted from RGB color spaces and morphological analysis
- Suitable for deployment on mobile and embedded devices
