# Quick Start Guide - RGB Model

## Prerequisites

1. **Python 3.8+** installed
2. **Kaggle account** (free)
3. **Git** (optional, for cloning)

## Setup (5 minutes)

### Step 1: Install Dependencies

```cmd
cd rgb_model
pip install -r requirements.txt
```

### Step 2: Configure Kaggle API

1. Go to https://www.kaggle.com/
2. Sign in → Account → "Create New API Token"
3. Download `kaggle.json`
4. Move to: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

## Quick Start - Automated Pipeline

### Option A: Run Complete Pipeline (Recommended)

```cmd
python run_pipeline.py
```

This runs everything automatically:

- Downloads dataset
- Preprocesses images
- Extracts features
- Trains all models
- Generates metrics and visualizations

**Time**: 30-60 minutes depending on dataset size

---

### Option B: Step-by-Step Execution

#### 1. Download Dataset

```cmd
cd data
python load_dataset.py
```

**Output**: Raw images in `data/raw/images/`

#### 2. Preprocess Data

```cmd
python preprocess.py
```

**Output**: Processed train/val/test splits in `data/processed/`

#### 3. Extract Features

```cmd
cd ..\features
python extract_features.py
```

**Output**: Feature vectors in `data/features/`

#### 4. Train Models

```cmd
cd ..\training
python train_rgb_model.py
```

**Output**:

- Trained models in `models/`
- Results in `models/training_results.json`
- Visualizations in `models/*.png`

---

## Test Your Model

### Single Prediction

```cmd
cd inference
python predict.py --image path\to\eye_image.jpg
```

### Compare All Models

```cmd
python predict.py --image path\to\eye_image.jpg --compare
```

### Save Result to JSON

```cmd
python predict.py --image path\to\eye_image.jpg --save result.json
```

---

## Expected Output

### Training Results

```
Model Performance on Test Set:
Model                     Accuracy   F1-Score   AUC-ROC    Time (ms)
-----------------------------------------------------------------
Random Forest             0.9200     0.8900     0.9300     12.34
Gradient Boosting         0.9300     0.9100     0.9400     18.56
Logistic Regression       0.8700     0.8500     0.8900     7.89
SVM                       0.9000     0.8800     0.9200     10.23
```

### Inference Output

```
📊 PREDICTION
  Class:      Anemia
  Confidence: 0.8700 (87.00%)

  Class Probabilities:
    Normal    : 0.1300 (13.00%)
    Anemia    : 0.8700 (87.00%)

⚙️  EXECUTION INFO
  Model:      random_forest
  OS:         Windows 10.0.19045

⏱️  TIMING BREAKDOWN
  Preprocessing:           45.23 ms
  Feature Extraction:      78.56 ms
  Inference:               12.34 ms
  ────────────────────────────────────────
  TOTAL:                  136.13 ms
```

---

## Troubleshooting

### Issue: "Kaggle API credentials not found"

**Solution**:

```cmd
# Check if kaggle.json exists
dir %USERPROFILE%\.kaggle\kaggle.json

# If not, download from Kaggle and place it there
```

### Issue: "Module not found" errors

**Solution**:

```cmd
pip install --upgrade -r requirements.txt
```

### Issue: Out of memory

**Solution**: Reduce image size in `data/preprocess.py`:

```python
preprocessor = ImagePreprocessor(target_size=(128, 128))
```

### Issue: Dataset download fails

**Solution**:

```cmd
# Test Kaggle API
kaggle datasets list

# If it works, retry download
python data/load_dataset.py
```

---

## Files Generated

After running the complete pipeline:

```
rgb_model/
├── data/
│   ├── raw/
│   │   └── images/          # Downloaded images
│   ├── processed/
│   │   ├── train/           # Training images (augmented)
│   │   ├── val/             # Validation images
│   │   └── test/            # Test images
│   └── features/
│       ├── train/           # Training features
│       ├── val/             # Validation features
│       ├── test/            # Test features
│       ├── scaler.pkl       # Feature scaler
│       └── feature_names.txt
│
└── models/
    ├── random_forest_model.pkl
    ├── gradient_boosting_model.pkl
    ├── logistic_regression_model.pkl
    ├── svm_model.pkl
    ├── training_results.json
    ├── confusion_matrices.png
    └── metrics_comparison.png
```

---

## Next Steps

1. ✅ **Review Results**: Check `models/training_results.json`
2. ✅ **Visualize Performance**: Open `models/*.png` files
3. ✅ **Test Inference**: Run predictions on sample images
4. ✅ **Deploy**: Export best model for mobile/embedded deployment
5. ✅ **Integrate**: Add personalization factors (family history, KIME, etc.)

---

## Tips for Best Results

1. **Dataset Quality**: Ensure eye images are well-lit and focused
2. **Augmentation**: Adjust augmentation factor in `preprocess.py` if dataset is small
3. **Feature Selection**: Experiment with different color spaces
4. **Model Selection**: Random Forest typically gives best accuracy/speed trade-off
5. **Hyperparameter Tuning**: Modify parameters in `train_rgb_model.py`

---

## Support

For issues or questions:

1. Check this guide
2. Review README.md for detailed documentation
3. Check error messages and stack traces
4. Open an issue in the repository

---

**Estimated Total Time**: 1-2 hours for complete setup and training

**Ready to Start?** Run: `python run_pipeline.py`
