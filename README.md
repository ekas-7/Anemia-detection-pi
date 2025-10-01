# Personalized Anemia Detection Framework

## 1. Personalization Factors

The system incorporates personalization layers to improve prediction accuracy:

- **Family History** → Genetic predispositions such as inherited hemoglobinopathies.
- **KIME (Key Individual Medical Evidence)** → Prior medical records, hemoglobin levels, and clinical history.
- **Dietary Patterns (Veg/Non-Veg)** → Nutritional deficiencies differ depending on dietary intake.
- **Sickle Cell Predisposition** → Identifying individuals with higher risk of anemia due to sickle cell or thalassemia traits.

---

## 2. Model Selection Based on Operating System (OS)

Depending on device capability and OS (Android, iOS, Desktop, Embedded):

- **RGB ML Model (Lightweight)**

  - For low-resource devices (mobile, embedded systems).
  - Input: Preprocessed RGB images.
  - Algorithm: Lightweight ML model (Random Forest, Logistic Regression, MobileNet-lite).
  - Goal: Quick, energy-efficient anemia screening.

- **CNN Model (Adaptive Performance)**
  - For high-performance devices (desktop, cloud servers, high-end mobile devices).
  - Input: Conjunctiva / eye-region images.
  - Architecture:
    - **MobileNet-based CNN**: For resource-constrained high-performance phones
    - **Deep CNN**: ResNet, EfficientNet, or U-Net for segmentation (desktop/cloud)
  - Goal: Detailed and accurate anemia classification with optimized performance.

---

## 3. Segmentation Pipeline

1. **Preprocessing**

   - Normalize RGB images
   - Enhance contrast & remove noise

2. **Segmentation**

   - Detect **eye conjunctiva** region
   - Models: U-Net / Mask R-CNN

3. **Feature Extraction**

   - Color histograms, vessel density, paleness index
   - Morphological features of conjunctiva

4. **Classification**
   - **If lightweight OS/device** → RGB ML Model
   - **If high-performance OS/device** → CNN Model

---

## 4. Datasets

- **Eyes Defy Anemia Dataset** → Eye images for anemia screening.
- **CP Anemia Dataset** → Clinical & parameter dataset.
- **Eye Conjunctiva Dataset** → Annotated conjunctival images with anemia severity levels.

---

## 5. Output & Applications

- **Severity Classification**: Normal / Mild / Moderate / Severe Anemia
- **Performance Metrics**:
  - **Accuracy**: Overall prediction accuracy percentage
  - **Confidence Score**: Model confidence level for each prediction
  - **F1-Score**: Harmonic mean of precision and recall
  - **Specificity**: True negative rate
  - **Sensitivity/Recall**: True positive rate
  - **Precision**: Positive predictive value
  - **Execution Time**: Time to run inference on target OS/device
  - **AUC-ROC**: Area under the receiver operating characteristic curve
- **Personalized Risk Profiling**: Combines image-based + clinical history features
- **Deployment Options**:
  - Mobile app (lightweight RGB ML model / MobileNet-based CNN)
  - Cloud API / Hospital system (deep CNN model)

---

## 6. Future Enhancements

- **Hybrid Ensemble**: Combine RGB + CNN predictions.
- **Cross-Platform Support**: Optimize for Android, iOS, Linux, Windows.
- **Multimodal Input**: Combine image + lab tests + family history for higher accuracy.

---

## Project Folder Structure

```
📂 anemia-detection/
│
├── 📂 cnn_segmentation_pipeline/ # Deep learning (CNN-based)
│ ├── 📂 data/ # Datasets (Eyes Defy, CP, Conjunctiva)
│ │ ├── eyes_defy_anemia/
│ │ ├── cp_anemia/
│ │ └── eye_conjunctiva/
│ │
│ ├── 📂 preprocessing/ # Normalization, noise removal
│ ├── 📂 segmentation/ # U-Net / Mask R-CNN
│ ├── 📂 models/ # CNN architectures (ResNet, EfficientNet)
│ ├── 📂 training/ # Training scripts, configs
│ ├── 📂 inference/ # Model serving scripts
│ └── README.md
│
├── 📂 rgb_model/ # Lightweight ML model (for low-power devices)
│ ├── 📂 data/ # Preprocessed RGB images
│ ├── 📂 features/ # Histogram, paleness index, etc.
│ ├── 📂 models/ # Random Forest / Logistic Regression
│ ├── 📂 training/
│ ├── 📂 inference/
│ └── README.md
│
├── 📂 flutter_app/ # Mobile app frontend
│ ├── 📂 lib/
│ │ ├── main.dart # Entry point
│ │ ├── screens/ # UI screens (upload, result, history)
│ │ ├── services/ # API calls to models
│ │ └── widgets/ # Reusable Flutter widgets
│ ├── 📂 assets/ # Icons, images
│ ├── 📂 test/ # Flutter tests
│ └── pubspec.yaml
│
├── 📂 docs/ # Documentation
│ ├── architecture.md
│ ├── datasets.md
│ ├── pipeline.md
│ └── personalization.md
│
└── README.md
```

---

### 📌 Notes:

- **cnn_segmentation_pipeline/** → heavy CNN models, best for cloud or high-performance systems.
- **rgb_model/** → lightweight ML model, optimized for mobile/embedded devices.
- **flutter_app/** → mobile frontend (Android/iOS) to interact with either model depending on OS & resources.

---

👉 Do you want me to also **add a deployment folder (docker, cloud, API)** so that both CNN and RGB models can be served through one API for the Flutter app?

If you confirm, I will create a `deployment/` folder with Docker, API, and cloud deployment examples. I will also create `README.md` files in the main subfolders now.
