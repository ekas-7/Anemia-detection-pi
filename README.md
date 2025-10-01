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

- **CNN Model (Heavyweight)**  
  - For high-performance devices (desktop, cloud servers).  
  - Input: Conjunctiva / eye-region images.  
  - Architecture: Deep CNN (ResNet, EfficientNet, or U-Net for segmentation).  
  - Goal: Detailed and accurate anemia classification.  

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
- **Personalized Risk Profiling**: Combines image-based + clinical history features  
- **Deployment Options**:  
  - Mobile app (lightweight RGB ML model)  
  - Cloud API / Hospital system (deep CNN model)  

---

## 6. Future Enhancements

- **Hybrid Ensemble**: Combine RGB + CNN predictions.  
- **Cross-Platform Support**: Optimize for Android, iOS, Linux, Windows.  
- **Multimodal Input**: Combine image + lab tests + family history for higher accuracy.  
