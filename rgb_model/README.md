# RGB Intensity-Based Anemia Detection# RGB Model - Lightweight Anemia Detection

## Overview## Overview

This module implements a **pure RGB intensity analysis** approach for anemia detection. Unlike the ML model which uses 113 extracted features and machine learning algorithms, this model focuses specifically on analyzing **RGB color intensity patterns** in conjunctiva images.This module contains a lightweight RGB-based machine learning model for anemia detection from eye images. It's optimized for low-resource devices like mobile phones and embedded systems.

## Methodology## Features

### Key Principle✅ **Multiple ML Algorithms**: Random Forest, Gradient Boosting, Logistic Regression, SVM

Anemia causes **reduced hemoglobin** in blood, which results in:✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, Specificity, AUC-ROC, Confidence Scores

- 🔴 **Lower red color intensity** in conjunctiva✅ **Fast Inference**: < 300ms on mobile devices

- 🤍 **Increased paleness** (higher overall brightness relative to red)✅ **Low Memory Footprint**: < 30 MB

- 📉 **Lower redness ratio** (R/G ratio)✅ **Color-Based Features**: RGB/HSV/LAB statistics, histograms, paleness index

✅ **Dataset Integration**: Automatic download from Kaggle (Eyes Defy Anemia)

### RGB Features Analyzed

## Project Structure

1. **Red Channel Intensity**

   - Mean, median, std, min, max of red channel```

   - Primary indicator of blood oxygenationrgb_model/

├── data/

2. **Redness Ratio (R/G)**│ ├── load_dataset.py # Download and load Eyes Defy Anemia dataset

   - Ratio of red to green intensity│ ├── preprocess.py # Image preprocessing and augmentation

   - Lower in anemic patients│ ├── raw/ # Raw downloaded data

│ ├── processed/ # Preprocessed train/val/test splits

3. **Paleness Score**│ └── features/ # Extracted features

   - Overall brightness / Red intensity│

   - Higher values indicate more paleness├── features/

│ └── extract_features.py # Color and morphological feature extraction

4. **Red Dominance**│

   - How much red channel dominates over other channels├── models/

   - Negative or low values suggest anemia│ └── (trained models saved here)

│

5. **Color Saturation**├── training/

   - How vivid/saturated the colors are│ └── train_rgb_model.py # Model training pipeline

   - Lower in pale, anemic conjunctiva│

├── inference/

## Model Architecture│ └── predict.py # Lightweight inference script

│

````├── requirements.txt           # Python dependencies

Input Image (224x224x3)└── README.md                  # This file

         ↓```

  RGB Channel Separation

         ↓## Installation

   Feature Extraction

   (10 RGB features)### 1. Prerequisites

         ↓

  Threshold-Based Classification- Python 3.8+

         ↓- pip

   Prediction + Confidence- Kaggle account (for dataset download)

````

### 2. Set up Kaggle API

**Decision Logic:**

- If 2+ indicators suggest anemia → Predict Anemia```bash

- Otherwise → Predict Normal# Install kaggle

pip install kaggle

## Training

# Set up credentials (Windows)

The model learns optimal thresholds from training data:# Copy kaggle.json to C:\Users\<YourUsername>\.kaggle\

- Calculates mean values for each class (Normal/Anemia)```

- Sets thresholds at midpoint between class means

- No complex ML algorithms needed!### 3. Install Dependencies

## Usage```bash

cd rgb_model

### Train the Modelpip install -r requirements.txt

````

```bash

cd rgb_model## Quick Start

python train_rgb_intensity.py

```### Step 1: Download Dataset



### Test on Single Image```bash

cd data

```bashpython load_dataset.py

python test_rgb_intensity.py --image path/to/image.png```

````

This will download the Eyes Defy Anemia dataset from Kaggle and organize images.

### Test on Multiple Images

### Step 2: Preprocess Data

````bash

python test_rgb_intensity.py --batch path/to/images/folder```bash

```python preprocess.py

````

## Example Output

Splits data into train/val/test, applies augmentation and preprocessing.

````

🔍 Prediction: Anemia### Step 3: Extract Features

📊 Confidence: 78.50%

📈 Anemia Score: 3.5 / 4.0```bash

cd ..\features

📋 Key RGB Intensity Features:python extract_features.py

  Red Mean Intensity:    87.45```

  Redness Ratio (R/G):   0.923

  Paleness Score:        1.457Extracts RGB/HSV/LAB color features, paleness index, and texture features.

  Red Dominance:         -8.32

  Overall Brightness:    94.21### Step 4: Train Models



⚠️  Anemia Indicators Detected:```bash

  • Low red intensity: 23.4% deviationcd ..\training

  • High paleness: 18.7% deviationpython train_rgb_model.py

  • Low redness ratio: 15.2% deviation```

````

Trains multiple ML models and evaluates with comprehensive metrics.

## Advantages

### Step 5: Run Inference

✅ **Simple & Interpretable**: Based on direct RGB intensity measurements

✅ **Fast**: No complex feature extraction or ML inference ```bash

✅ **Lightweight**: < 1KB model size (just thresholds) cd ..\inference

✅ **Explainable**: Shows exactly which RGB features indicate anemia

✅ **No Training Needed**: Can work with just threshold tuning # Single model prediction

python predict.py --image path\to\eye_image.jpg --model random_forest

## Disadvantages

# Compare all models

⚠️ May be sensitive to:python predict.py --image path\to\eye_image.jpg --compare

- Lighting conditions

- Camera white balance# Save result to JSON

- Image qualitypython predict.py --image path\to\eye_image.jpg --save result.json

- Skin tone variations```

## Files## Performance Metrics

````The model outputs comprehensive metrics including:

rgb_model/

├── train_rgb_intensity.py      # Training script### Classification Metrics

├── test_rgb_intensity.py        # Inference script

├── models/- **Accuracy**: Overall prediction accuracy

│   ├── rgb_intensity_model.json # Trained thresholds- **Precision**: Positive predictive value

│   ├── rgb_intensity_results.json- **Recall/Sensitivity**: True positive rate

│   └── rgb_intensity_results.png- **Specificity**: True negative rate

└── README.md                    # This file- **F1-Score**: Harmonic mean of precision and recall

```- **AUC-ROC**: Area under ROC curve



## Performance Metrics### Confidence Metrics



After training, the model provides:- **Confidence Score**: Model confidence for each prediction (0-1)

- Accuracy- **Class Probabilities**: Probability distribution across classes

- Precision, Recall, F1-Score

- Sensitivity, Specificity### Execution Metrics

- AUC-ROC

- Confusion Matrix- **Preprocessing Time**: Image loading and preprocessing (ms)

- Confidence score distribution- **Feature Extraction Time**: Time to extract features (ms)

- **Inference Time**: Model prediction time (ms)

## Comparison with ML Model- **Total Time**: End-to-end execution time (ms)



| Aspect | RGB Intensity Model | ML Model |## Example Output

|--------|-------------------|----------|

| Features | 10 RGB features | 113 color/texture features |```json

| Algorithm | Threshold-based | Random Forest, GBM, SVM, LR |{

| Model Size | < 1 KB | 1-6 MB |  "prediction": {

| Inference | ~50-100ms | ~100-300ms |    "class": "Anemia",

| Interpretability | High | Medium |    "confidence_score": 0.87,

| Accuracy | Moderate | Higher |    "class_probabilities": {

      "Normal": 0.13,

## Requirements      "Anemia": 0.87

    }

```  },

numpy  "execution_info": {

opencv-python    "model_type": "random_forest",

matplotlib    "os": "Windows",

seaborn    "inference_time_ms": 12.34,

scikit-learn    "total_time_ms": 136.13

```  }

}

## Citation```



Based on medical literature:## Dataset

- Anemia reduces hemoglobin → Less red color in blood vessels

- Conjunctiva examination is a standard clinical method for anemia screening**Eyes Defy Anemia Dataset** from Kaggle:

- RGB intensity analysis correlates with hemoglobin levels

- Dataset ID: `harshwardhanfartale/eyes-defy-anemia`
- Contains eye images labeled for anemia detection
- Automatically downloaded via `load_dataset.py`

## Notes

- Optimized for low-power devices and fast inference
- Models saved in `models/` directory
- Features extracted from RGB color spaces and morphological analysis
- Suitable for deployment on mobile and embedded devices
````
