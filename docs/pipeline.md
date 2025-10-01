# Pipeline

## Overview

The anemia detection pipeline consists of multiple stages, from data preprocessing to model inference and performance evaluation. The pipeline adapts based on the target device and model architecture (RGB ML, MobileNet CNN, or Deep CNN).

---

## 1. Preprocessing

### Image Acquisition

- **Sources**: Mobile camera, clinical imaging systems, uploaded images
- **Formats**: JPEG, PNG, HEIC
- **Resolution**: Variable (auto-adjusted based on model requirements)

### Preprocessing Steps

#### A. Image Validation

- Check image quality (blur detection, lighting conditions)
- Verify presence of eye region
- Reject low-quality images

#### B. Normalization

- Resize images to model-specific dimensions:
  - RGB ML: 128x128 or 224x224
  - MobileNet CNN: 224x224
  - Deep CNN: 512x512 or 1024x1024
- Pixel value normalization: [0, 255] → [0, 1] or [-1, 1]

#### C. Contrast Enhancement

- Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Enhance color channel separation for better feature visibility

#### D. Noise Reduction

- Gaussian blur or bilateral filtering
- Remove artifacts from image compression

#### E. Color Space Conversion

- Convert RGB to HSV/LAB for advanced color analysis
- Extract color channel statistics

---

## 2. Segmentation

### Purpose

Isolate the conjunctiva region (eye white area) for focused anemia detection.

### Segmentation Models

#### A. Lightweight Segmentation (Mobile Devices)

- **Model**: MobileNet-based segmentation or classical CV techniques
- **Method**:
  - Eye detection using Haar Cascades or MediaPipe
  - ROI extraction around detected eyes
  - Simple thresholding for conjunctiva isolation

#### B. Deep Learning Segmentation (High-Performance Devices)

- **Model**: U-Net or Mask R-CNN
- **Architecture**:
  - Encoder: MobileNetV2 (mobile) or ResNet-50 (desktop)
  - Decoder: Upsampling layers with skip connections
- **Output**: Binary mask highlighting conjunctiva region

### Segmentation Pipeline

1. Detect eye region using face/eye detection
2. Extract bounding box around eyes
3. Apply segmentation model to isolate conjunctiva
4. Post-process mask (morphological operations)
5. Apply mask to original image for feature extraction

---

## 3. Feature Extraction

### Color Features

- **RGB Channel Statistics**: Mean, median, standard deviation per channel
- **HSV Analysis**: Hue, saturation, value distributions
- **LAB Color Space**: Lightness, a* (green-red), b* (blue-yellow)
- **Color Histograms**: 256-bin histograms for each channel

### Morphological Features

- **Paleness Index**: Computed from red channel intensity
- **Vessel Density**: Blood vessel prominence in conjunctiva
- **Texture Features**: GLCM (Gray-Level Co-occurrence Matrix) features
- **Edge Density**: Canny edge detection statistics

### Clinical Features (from Personalization Layer)

- Family history encoding
- KIME score
- Dietary pattern (veg/non-veg binary or categorical)
- Sickle cell predisposition score

---

## 4. Model Training

### RGB ML Model Training

#### Dataset Preparation

- Split: 70% train, 15% validation, 15% test
- Augmentation: Rotation, flipping, brightness adjustment, color jitter

#### Training Process

- **Algorithms**: Random Forest, Gradient Boosting, Logistic Regression
- **Features**: Extracted color and morphological features
- **Hyperparameter Tuning**: Grid search or Bayesian optimization
- **Validation**: Cross-validation for robustness

### MobileNet CNN Training (Resource-Constrained High-Performance Devices)

#### Architecture

- **Backbone**: MobileNetV2 or MobileNetV3
- **Modifications**:
  - Replace final classification layer
  - Add dropout layers for regularization
  - Optional: Add attention mechanisms

#### Training Configuration

- **Framework**: TensorFlow Lite / PyTorch Mobile
- **Loss Function**: Categorical cross-entropy
- **Optimizer**: Adam with learning rate scheduling
- **Batch Size**: 32-64
- **Epochs**: 50-100 with early stopping
- **Data Augmentation**: Random rotation, flip, zoom, brightness, contrast

#### Transfer Learning

- Start with ImageNet pre-trained weights
- Fine-tune on anemia detection dataset
- Freeze early layers, train later layers

#### Quantization

- Post-training quantization to reduce model size
- INT8 quantization for faster inference
- Validate accuracy after quantization

### Deep CNN Training (Desktop/Cloud)

#### Architecture Options

- **ResNet-50/101**: Residual connections for deep networks
- **EfficientNet-B0 to B7**: Compound scaling for efficiency
- **Custom U-Net**: For segmentation + classification pipeline

#### Training Configuration

- **Framework**: PyTorch or TensorFlow
- **Loss Function**: Weighted cross-entropy (to handle class imbalance)
- **Optimizer**: AdamW or SGD with momentum
- **Batch Size**: 16-32 (depending on GPU memory)
- **Epochs**: 100-200 with early stopping
- **Learning Rate**: 1e-4 to 1e-3 with cosine annealing

#### Advanced Techniques

- **Mixed Precision Training**: FP16 for faster training
- **Gradient Accumulation**: Simulate larger batch sizes
- **Ensemble Learning**: Combine multiple model predictions
- **Multi-task Learning**: Joint segmentation and classification

---

## 5. Inference Pipeline

### Device-Specific Inference

#### Mobile Inference (RGB ML / MobileNet CNN)

1. Capture or upload image
2. Run lightweight preprocessing
3. Load optimized model (TFLite / Core ML / ONNX)
4. Perform inference (< 300ms)
5. Post-process output
6. Display results with metrics

#### Desktop/Cloud Inference (Deep CNN)

1. Receive high-resolution image
2. Run full preprocessing pipeline
3. Load deep CNN model (with GPU)
4. Perform inference (300ms - 2s)
5. Generate detailed outputs with all metrics
6. Return results via API or display

### Inference Output Structure

```json
{
  "prediction": {
    "class": "Moderate Anemia",
    "severity_level": 2,
    "confidence_score": 0.87
  },
  "performance_metrics": {
    "accuracy": 0.92,
    "f1_score": 0.89,
    "precision": 0.91,
    "recall": 0.88,
    "specificity": 0.94,
    "auc_roc": 0.93
  },
  "execution_info": {
    "model_type": "MobileNet_CNN",
    "os": "Android 13",
    "device": "Pixel 7 Pro",
    "inference_time_ms": 245,
    "preprocessing_time_ms": 120
  },
  "personalization": {
    "family_history_risk": "High",
    "dietary_factor": "Vegetarian",
    "kime_score": 0.65,
    "sickle_cell_risk": "Low",
    "adjusted_risk_score": 0.78
  },
  "recommendations": [
    "Consult a healthcare provider for blood test confirmation",
    "Consider iron supplementation",
    "Increase iron-rich food intake"
  ]
}
```

---

## 6. Performance Evaluation

### Metrics Computed

#### Classification Metrics

- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall/Sensitivity**: TP / (TP + FN)
- **Specificity**: TN / (TN + FP)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **AUC-ROC**: Area under receiver operating characteristic curve

#### Per-Class Metrics

- Individual precision, recall, F1-score for each severity class
- Confusion matrix visualization

#### Execution Metrics

- **Inference Time**: Time to run model forward pass
- **Preprocessing Time**: Time for image processing
- **Total Pipeline Time**: End-to-end execution time
- **Memory Usage**: Peak memory consumption during inference
- **Battery Impact**: (For mobile devices) Power consumption estimate

### Evaluation Protocol

1. **Hold-out Test Set**: 15% of dataset never seen during training
2. **Cross-Validation**: 5-fold or 10-fold for robust evaluation
3. **Stratified Sampling**: Ensure balanced representation of severity levels
4. **Real-world Testing**: Evaluate on clinical data from hospitals
5. **Device-specific Testing**: Test on target hardware (mobile, desktop, cloud)

### Benchmarking

- Compare RGB ML vs MobileNet CNN vs Deep CNN
- Evaluate trade-offs: Accuracy vs Speed vs Resource Usage
- Test on different OS: Android, iOS, Windows, Linux, macOS
- Profile inference time on various device tiers

---

## 7. Model Optimization

### For Mobile Deployment

- **Quantization**: Convert FP32 to INT8
- **Pruning**: Remove redundant weights
- **Knowledge Distillation**: Train smaller model to mimic larger one
- **Layer Fusion**: Combine consecutive operations

### For Cloud Deployment

- **Batching**: Process multiple images simultaneously
- **Caching**: Store preprocessed features
- **Model Serving**: Use TensorFlow Serving, TorchServe, or ONNX Runtime
- **Auto-scaling**: Scale instances based on load

---

## 8. Continuous Improvement

### Model Updates

- Retrain with new data periodically
- A/B testing for model versions
- Collect feedback from users and clinicians
- Address edge cases and failure modes

### Monitoring

- Track inference metrics in production
- Monitor model drift and performance degradation
- Log misclassifications for review
- Implement feedback loop for continuous learning
