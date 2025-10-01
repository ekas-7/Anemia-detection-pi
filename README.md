# Personalized Anemia Detection Framework

## 1. Personalization Factors
The model will incorporate personalization layers to improve prediction accuracy:
- **Based on OS (Occupation/Socioeconomic)**  
  Risk factors vary depending on lifestyle, healthcare access, and occupational exposure.  
- **Based on Family History**  
  Genetic predispositions such as inherited hemoglobinopathies.  
- **KIME (Key Individual Medical Evidence)**  
  Incorporating prior medical records, hemoglobin levels, and clinical history.  
- **Dietary Patterns (Veg/Non-Veg)**  
  Nutritional deficiencies differ depending on dietary intake.  
- **Sickle Cell Predisposition**  
  Identifying individuals with higher risk of anemia due to sickle cell or thalassemia traits.  

---

## 2. Model Approaches

### Model 1: RGB-Based Imaging Model  
- Input: Conjunctiva (eye images), palm, tongue photographs (RGB format).  
- Architecture: **CNN (Convolutional Neural Network)** trained on preprocessed images.  
- Goal: Identify pale conjunctiva, discoloration patterns, and biomarkers of anemia.  

### Model 2: ML Model with Clinical Features  
- Input: Patient history, family data, lab parameters, dietary habits.  
- Architecture: Classical ML models (Random Forest, XGBoost) or lightweight neural nets.  
- Goal: Predict likelihood of anemia and type (iron-deficiency, sickle-cell-related, chronic disease).  

---

## 3. Segmentation Pipeline

1. **Preprocessing**  
   - RGB image normalization  
   - Contrast enhancement  
   - Noise reduction  

2. **Segmentation**  
   - Localize **eye conjunctiva** (primary biomarker region)  
   - U-Net / Mask R-CNN for precise boundary detection  

3. **Feature Extraction**  
   - Color histograms, texture features  
   - Vessel density analysis  
   - Redness/paleness indices  

4. **Classification**  
   - CNN → classify as *normal / mild / severe anemia*  
   - Ensemble with ML clinical model for hybrid decision-making  

---

## 4. Datasets

- **Eyes Defy Anemia Dataset**  
  Images of eyes for anemia screening.  
- **CP Anemia Dataset**  
  Clinical parameter dataset for anemia classification.  
- **Eye Conjunctiva Dataset**  
  Specific dataset focusing on conjunctival images annotated for anemia severity.  

---

## 5. Output & Applications

- **Severity Classification**: Normal, Mild, Moderate, Severe Anemia  
- **Personalized Risk Profiling**: Combining image + clinical history  
- **Deployment**: Mobile app for low-resource settings, cloud-based API for hospitals  

---

## 6. Future Directions

- Expand datasets across diverse ethnicities and age groups.  
- Integrate multimodal data (eye images + blood tests + EHR).  
- Explore **transfer learning** from medical vision models (e.g., MedCLIP, ViT).  
- Deploy in **edge devices** for rural healthcare.  
