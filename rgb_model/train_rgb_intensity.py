"""
RGB Intensity-Based Anemia Detection Model
============================================================
This model analyzes RGB color intensity values to detect anemia.
It focuses on:
1. Red channel intensity (lower in anemia)
2. Paleness index (higher in anemia)
3. Redness ratio (R/G ratio)
4. Overall color intensity patterns
"""

import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

class RGBIntensityAnalyzer:
    """Analyzes RGB intensity for anemia detection"""
    
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.thresholds = {
            'red_intensity': None,      # Will be learned from training data
            'paleness_score': None,
            'redness_ratio': None,
            'brightness': None
        }
        
    def load_image(self, image_path):
        """Load and preprocess image"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, self.image_size)
        
        return img
    
    def extract_rgb_intensity_features(self, image):
        """Extract RGB intensity-based features for anemia detection"""
        
        # Separate RGB channels
        R = image[:, :, 0]
        G = image[:, :, 1]
        B = image[:, :, 2]
        
        features = {}
        
        # 1. Red Channel Intensity (primary indicator)
        features['red_mean'] = np.mean(R)
        features['red_median'] = np.median(R)
        features['red_std'] = np.std(R)
        features['red_max'] = np.max(R)
        features['red_min'] = np.min(R)
        
        # 2. Green Channel Intensity
        features['green_mean'] = np.mean(G)
        features['green_median'] = np.median(G)
        
        # 3. Blue Channel Intensity
        features['blue_mean'] = np.mean(B)
        features['blue_median'] = np.median(B)
        
        # 4. Redness Ratio (R/G ratio - lower in anemia)
        features['redness_ratio'] = features['red_mean'] / (features['green_mean'] + 1e-6)
        
        # 5. Red-Blue Ratio
        features['red_blue_ratio'] = features['red_mean'] / (features['blue_mean'] + 1e-6)
        
        # 6. Paleness Score (higher values = more pale = anemia)
        # Paleness is indicated by low red intensity and high overall brightness
        overall_brightness = (features['red_mean'] + features['green_mean'] + features['blue_mean']) / 3
        features['paleness_score'] = overall_brightness / (features['red_mean'] + 1e-6)
        
        # 7. Overall Brightness
        features['brightness'] = overall_brightness
        
        # 8. Red Intensity (percentage of maximum)
        features['red_intensity_percent'] = (features['red_mean'] / 255.0) * 100
        
        # 9. Color Saturation (how vivid the color is)
        max_rgb = np.maximum(np.maximum(R, G), B)
        min_rgb = np.minimum(np.minimum(R, G), B)
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
        features['saturation_mean'] = np.mean(saturation)
        
        # 10. Red dominance (how much red dominates over other channels)
        features['red_dominance'] = features['red_mean'] - (features['green_mean'] + features['blue_mean']) / 2
        
        return features
    
    def predict_from_features(self, features):
        """
        Predict anemia based on RGB intensity features
        
        Anemia indicators:
        - Low red intensity (< threshold)
        - High paleness score (> threshold)
        - Low redness ratio (< threshold)
        - Lower overall brightness in red channel
        """
        
        if self.thresholds['red_intensity'] is None:
            raise ValueError("Model not trained! Call train() first.")
        
        # Score calculation (0 = normal, higher = more anemic)
        anemia_score = 0
        confidence_factors = []
        
        # Check red intensity (lower = anemia)
        if features['red_mean'] < self.thresholds['red_intensity']:
            anemia_score += 1
            confidence_factors.append(('Low red intensity', 
                                      (self.thresholds['red_intensity'] - features['red_mean']) / self.thresholds['red_intensity']))
        
        # Check paleness score (higher = anemia)
        if features['paleness_score'] > self.thresholds['paleness_score']:
            anemia_score += 1
            confidence_factors.append(('High paleness', 
                                      (features['paleness_score'] - self.thresholds['paleness_score']) / self.thresholds['paleness_score']))
        
        # Check redness ratio (lower = anemia)
        if features['redness_ratio'] < self.thresholds['redness_ratio']:
            anemia_score += 1
            confidence_factors.append(('Low redness ratio', 
                                      (self.thresholds['redness_ratio'] - features['redness_ratio']) / self.thresholds['redness_ratio']))
        
        # Check red dominance (lower = anemia)
        if features['red_dominance'] < 0:
            anemia_score += 0.5
            confidence_factors.append(('Weak red dominance', abs(features['red_dominance']) / 50))
        
        # Prediction (if 2 or more indicators suggest anemia)
        prediction = 1 if anemia_score >= 2 else 0
        
        # Confidence calculation
        confidence = min(anemia_score / 4.0, 1.0) if prediction == 1 else min((4 - anemia_score) / 4.0, 1.0)
        
        return {
            'prediction': prediction,
            'prediction_label': 'Anemia' if prediction == 1 else 'Normal',
            'confidence': confidence,
            'anemia_score': anemia_score,
            'features': features,
            'confidence_factors': confidence_factors
        }
    
    def train(self, X_train, y_train):
        """
        Learn optimal thresholds from training data
        Uses statistical analysis to find best separation points
        """
        
        print("\n" + "="*60)
        print("Training RGB Intensity Model")
        print("="*60)
        
        # Separate normal and anemia samples
        normal_features = [f for f, label in zip(X_train, y_train) if label == 0]
        anemia_features = [f for f, label in zip(X_train, y_train) if label == 1]
        
        print(f"\nTraining samples:")
        print(f"  Normal: {len(normal_features)}")
        print(f"  Anemia: {len(anemia_features)}")
        
        # Calculate mean values for each class
        normal_red_mean = np.mean([f['red_mean'] for f in normal_features])
        anemia_red_mean = np.mean([f['red_mean'] for f in anemia_features])
        
        normal_paleness = np.mean([f['paleness_score'] for f in normal_features])
        anemia_paleness = np.mean([f['paleness_score'] for f in anemia_features])
        
        normal_redness = np.mean([f['redness_ratio'] for f in normal_features])
        anemia_redness = np.mean([f['redness_ratio'] for f in anemia_features])
        
        # Set thresholds at midpoint between normal and anemia means
        self.thresholds['red_intensity'] = (normal_red_mean + anemia_red_mean) / 2
        self.thresholds['paleness_score'] = (normal_paleness + anemia_paleness) / 2
        self.thresholds['redness_ratio'] = (normal_redness + anemia_redness) / 2
        
        print(f"\nLearned Thresholds:")
        print(f"  Red Intensity: {self.thresholds['red_intensity']:.2f}")
        print(f"    Normal avg: {normal_red_mean:.2f}, Anemia avg: {anemia_red_mean:.2f}")
        print(f"  Paleness Score: {self.thresholds['paleness_score']:.2f}")
        print(f"    Normal avg: {normal_paleness:.2f}, Anemia avg: {anemia_paleness:.2f}")
        print(f"  Redness Ratio: {self.thresholds['redness_ratio']:.2f}")
        print(f"    Normal avg: {normal_redness:.2f}, Anemia avg: {anemia_redness:.2f}")
        
        return self
    
    def save_model(self, filepath):
        """Save model thresholds"""
        # Convert numpy types to native Python types
        thresholds_serializable = {k: float(v) if v is not None else None 
                                  for k, v in self.thresholds.items()}
        
        model_data = {
            'thresholds': thresholds_serializable,
            'image_size': list(self.image_size)
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"\n✓ Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load model thresholds"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.thresholds = model_data['thresholds']
        self.image_size = tuple(model_data['image_size'])
        
        print(f"✓ Model loaded from: {filepath}")
        return self


def load_dataset(data_dir):
    """Load images and extract features"""
    
    data_dir = Path(data_dir)
    
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)
    
    # Load from ml_model processed data
    # Get the project root and navigate to ml_model
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / 'ml_model' / 'data' / 'processed'
    
    if not processed_dir.exists():
        raise ValueError(f"Processed data not found at {processed_dir}")
    
    # Load numpy arrays
    X_train = np.load(processed_dir / 'train' / 'images.npy')
    y_train = np.load(processed_dir / 'train' / 'labels.npy')
    
    X_val = np.load(processed_dir / 'val' / 'images.npy')
    y_val = np.load(processed_dir / 'val' / 'labels.npy')
    
    X_test = np.load(processed_dir / 'test' / 'images.npy')
    y_test = np.load(processed_dir / 'test' / 'labels.npy')
    
    print(f"\nDataset loaded:")
    print(f"  Train: {len(X_train)} images")
    print(f"  Val:   {len(X_val)} images")
    print(f"  Test:  {len(X_test)} images")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def extract_features_from_dataset(images, analyzer):
    """Extract RGB intensity features from all images"""
    
    features_list = []
    
    print(f"\nExtracting RGB intensity features from {len(images)} images...")
    
    for img in images:
        features = analyzer.extract_rgb_intensity_features(img)
        features_list.append(features)
    
    print("✓ Feature extraction complete")
    
    return features_list


def evaluate_model(analyzer, X_test_features, y_test):
    """Evaluate model performance"""
    
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)
    
    # Make predictions
    predictions = []
    confidences = []
    
    for features in X_test_features:
        result = analyzer.predict_from_features(features)
        predictions.append(result['prediction'])
        confidences.append(result['confidence'])
    
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall
    
    # AUC-ROC
    try:
        auc_roc = roc_auc_score(y_test, confidences)
    except:
        auc_roc = 0.5
    
    # Print results
    print(f"\n{'='*60}")
    print("Test Set Results")
    print(f"{'='*60}")
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"Sensitivity:   {sensitivity:.4f}")
    print(f"Specificity:   {specificity:.4f}")
    print(f"F1-Score:      {f1:.4f}")
    print(f"AUC-ROC:       {auc_roc:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:4d}  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  TP: {tp:4d}")
    
    print(f"\nConfidence Scores:")
    print(f"  Mean:   {np.mean(confidences):.4f}")
    print(f"  Median: {np.median(confidences):.4f}")
    print(f"  Range:  [{np.min(confidences):.4f}, {np.max(confidences):.4f}]")
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'auc_roc': float(auc_roc),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'confidence_stats': {
            'mean': float(np.mean(confidences)),
            'median': float(np.median(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences))
        }
    }
    
    return results, cm, predictions, confidences


def plot_results(cm, y_test, predictions, confidences, save_dir):
    """Create visualization plots"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix
    ax = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    ax.set_xticklabels(['Normal', 'Anemia'])
    ax.set_yticklabels(['Normal', 'Anemia'])
    
    # 2. Confidence Distribution
    ax = axes[0, 1]
    ax.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Confidence Score Distribution')
    ax.axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.3f}')
    ax.legend()
    
    # 3. Prediction Distribution
    ax = axes[1, 0]
    unique, counts = np.unique(predictions, return_counts=True)
    labels = ['Normal', 'Anemia']
    colors = ['green', 'red']
    ax.bar([labels[i] for i in unique], counts, color=[colors[i] for i in unique], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Distribution')
    for i, (u, c) in enumerate(zip(unique, counts)):
        ax.text(i, c + 2, str(c), ha='center', fontweight='bold')
    
    # 4. Actual vs Predicted
    ax = axes[1, 1]
    correct = predictions == y_test
    incorrect = ~correct
    ax.scatter(np.where(correct)[0], confidences[correct], c='green', alpha=0.6, label='Correct', s=50)
    ax.scatter(np.where(incorrect)[0], confidences[incorrect], c='red', alpha=0.6, label='Incorrect', s=50, marker='x')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Confidence')
    ax.set_title('Prediction Confidence (Correct vs Incorrect)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'rgb_intensity_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plots saved to: {save_dir / 'rgb_intensity_results.png'}")
    plt.close()


def main():
    """Main training and evaluation pipeline"""
    
    print("="*60)
    print("RGB INTENSITY-BASED ANEMIA DETECTION")
    print("="*60)
    
    # Initialize analyzer
    analyzer = RGBIntensityAnalyzer(image_size=(224, 224))
    
    # Load dataset
    data_dir = Path('data')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(data_dir)
    
    # Extract RGB intensity features
    print("\n" + "="*60)
    print("Feature Extraction")
    print("="*60)
    
    X_train_features = extract_features_from_dataset(X_train, analyzer)
    X_val_features = extract_features_from_dataset(X_val, analyzer)
    X_test_features = extract_features_from_dataset(X_test, analyzer)
    
    # Train model (learn thresholds)
    analyzer.train(X_train_features, y_train)
    
    # Evaluate on validation set
    print("\n" + "="*60)
    print("Validation Set Evaluation")
    print("="*60)
    
    val_results, _, _, _ = evaluate_model(analyzer, X_val_features, y_val)
    
    # Evaluate on test set
    test_results, cm, predictions, confidences = evaluate_model(analyzer, X_test_features, y_test)
    
    # Save model
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    analyzer.save_model(model_dir / 'rgb_intensity_model.json')
    
    # Save results (convert all numpy types to native Python types)
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results_data = {
        'validation': convert_to_native(val_results),
        'test': convert_to_native(test_results),
        'thresholds': convert_to_native(analyzer.thresholds)
    }
    
    with open(model_dir / 'rgb_intensity_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {model_dir / 'rgb_intensity_results.json'}")
    
    # Create visualizations
    plot_results(cm, y_test, predictions, confidences, model_dir)
    
    print("\n" + "="*60)
    print("✓ RGB Intensity Model Training Complete!")
    print("="*60)
    print(f"\nTest Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test F1-Score: {test_results['f1_score']:.4f}")


if __name__ == "__main__":
    main()
