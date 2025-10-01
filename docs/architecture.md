# Architecture

## System Overview

The Personalized Anemia Detection Framework is designed with adaptive model selection based on device capabilities and operating system constraints.

## Components

### 1. Model Selection Layer

- **Device Detection**: Automatically detects OS, hardware capabilities, and available resources
- **Model Router**: Routes inference to appropriate model based on device performance tier

### 2. Model Architectures

#### RGB ML Model (Lightweight)

- **Target Devices**: Low-resource mobile devices, embedded systems
- **Algorithms**: Random Forest, Logistic Regression, MobileNet-lite
- **Input**: Preprocessed RGB images (224x224 or 128x128)
- **Memory Footprint**: < 10 MB
- **Inference Time**: < 100ms on mobile devices

#### CNN Model with MobileNet (Optimized for High-Performance Mobile)

- **Target Devices**: High-performance smartphones (flagship Android/iOS devices)
- **Architecture**: MobileNetV2/V3 backbone with custom classification head
- **Key Features**:
  - Depthwise separable convolutions for efficiency
  - Inverted residual blocks
  - Linear bottlenecks
- **Input**: Conjunctiva/eye-region images (224x224)
- **Memory Footprint**: 10-30 MB
- **Inference Time**: 100-300ms on high-end mobile devices
- **Optimization**: TensorFlow Lite / PyTorch Mobile quantization

#### Deep CNN Model (High-Performance Desktop/Cloud)

- **Target Devices**: Desktop workstations, cloud servers, medical systems
- **Architecture Options**:
  - ResNet-50/101 for classification
  - EfficientNet-B0 to B7 (scalable)
  - U-Net for segmentation tasks
- **Input**: High-resolution conjunctiva images (512x512 or higher)
- **Memory Footprint**: 100-500 MB
- **Inference Time**: 300ms - 2s depending on architecture
- **GPU Acceleration**: CUDA support for NVIDIA GPUs

### 3. Preprocessing Pipeline

- Image normalization and resizing
- Contrast enhancement (CLAHE)
- Noise reduction
- Color space conversion (RGB to HSV/LAB for feature extraction)

### 4. Segmentation Module

- Eye region detection
- Conjunctiva isolation using U-Net or Mask R-CNN
- Region of interest (ROI) extraction

### 5. Feature Extraction

- Color histograms (RGB, HSV channels)
- Paleness index calculation
- Vessel density metrics
- Morphological features

### 6. Personalization Layer

- Family history integration
- KIME (Key Individual Medical Evidence) incorporation
- Dietary pattern analysis
- Sickle cell predisposition scoring

### 7. Output Module

- Severity classification (Normal/Mild/Moderate/Severe)
- Performance metrics generation:
  - Accuracy
  - Confidence score
  - F1-score
  - Specificity
  - Sensitivity/Recall
  - Precision
  - Execution time (OS-specific)
  - AUC-ROC

## Data Flow

1. **Image Acquisition** → Camera/Upload
2. **Device Profiling** → OS detection, hardware capability assessment
3. **Preprocessing** → Normalization, enhancement, segmentation
4. **Model Selection** → Route to RGB ML / MobileNet CNN / Deep CNN
5. **Inference** → Run selected model
6. **Personalization** → Apply clinical factors and history
7. **Output Generation** → Classification + metrics + risk profiling
8. **Result Delivery** → Display with confidence scores and performance metrics

## API Endpoints (For Cloud Deployment)

### POST /predict

- **Input**: Image file, device_info, patient_metadata
- **Output**: JSON with classification, metrics, and personalized risk score

### GET /metrics

- **Output**: Model performance statistics on current dataset

### POST /personalize

- **Input**: Patient history, dietary info, family history
- **Output**: Adjusted risk profile

## Deployment Options

### Mobile Deployment

- **Android**: TensorFlow Lite / ONNX Runtime
- **iOS**: Core ML / PyTorch Mobile
- **Model Selection**: RGB ML or MobileNet CNN based on device tier

### Desktop Deployment

- **Windows/Linux/macOS**: Full Python environment with PyTorch/TensorFlow
- **Model**: Deep CNN with GPU acceleration

### Cloud Deployment

- **AWS/GCP/Azure**: Containerized model serving
- **API Gateway**: RESTful API for inference requests
- **Model**: Deep CNN with auto-scaling

### Embedded Systems

- **Raspberry Pi/Jetson Nano**: RGB ML model with optimized inference
- **Edge TPU**: Quantized MobileNet for accelerated edge inference
