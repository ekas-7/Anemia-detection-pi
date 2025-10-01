"""
Enhanced PKL Model Viewer - Using joblib
============================================================
This script loads and displays comprehensive information from 
the trained models using joblib (scikit-learn's preferred method)
"""

import joblib
import numpy as np
import json
from pathlib import Path
import sys

def load_model(filepath):
    """Load a model using joblib"""
    try:
        return joblib.load(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def inspect_model(model, model_name):
    """Display detailed information about a model"""
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")
    
    print(f"\n📦 Model Type: {type(model).__name__}")
    print(f"📦 Module: {type(model).__module__}")
    
    # Get model parameters
    if hasattr(model, 'get_params'):
        params = model.get_params()
        print(f"\n⚙️  Model Parameters ({len(params)} total):")
        important_params = ['n_estimators', 'max_depth', 'learning_rate', 'C', 'kernel', 
                          'max_features', 'min_samples_split', 'min_samples_leaf']
        for key in important_params:
            if key in params:
                print(f"   • {key}: {params[key]}")
    
    # Model-specific information
    if hasattr(model, 'n_features_in_'):
        print(f"\n📊 Number of features: {model.n_features_in_}")
    
    if hasattr(model, 'classes_'):
        print(f"📊 Classes: {model.classes_} (0=Normal, 1=Anemia)")
    
    if hasattr(model, 'feature_importances_'):
        print(f"\n⭐ Top 15 Feature Importances:")
        importances = model.feature_importances_
        
        # Load feature names
        try:
            feature_names_path = Path("data/features/feature_names.txt")
            with open(feature_names_path, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
        except:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        indices = np.argsort(importances)[::-1]
        for i in range(min(15, len(importances))):
            idx = indices[i]
            feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            print(f"   {i+1:2d}. {feat_name:<30s}: {importances[idx]:.6f}")
    
    if hasattr(model, 'n_estimators'):
        print(f"\n🌳 Number of estimators: {model.n_estimators}")
    
    if hasattr(model, 'estimators_') and hasattr(model, 'n_estimators'):
        if isinstance(model.estimators_, np.ndarray):
            print(f"🌳 Trees trained: {len(model.estimators_)}")
        elif isinstance(model.estimators_, list):
            print(f"🌳 Estimators trained: {len(model.estimators_)}")
    
    if hasattr(model, 'coef_'):
        print(f"\n📐 Model coefficients shape: {model.coef_.shape}")
        if model.coef_.shape[1] <= 10:
            print(f"   Coefficients: {model.coef_[0]}")
        else:
            print(f"   First 10 coefficients: {model.coef_[0][:10]}")
            print(f"   Last 10 coefficients: {model.coef_[0][-10:]}")
    
    if hasattr(model, 'intercept_'):
        print(f"📐 Intercept: {model.intercept_}")
    
    if hasattr(model, 'support_vectors_'):
        print(f"\n🎯 Number of support vectors: {len(model.support_vectors_)}")
    
    # Memory size estimate
    try:
        import pickle
        size_bytes = len(pickle.dumps(model))
        size_kb = size_bytes / 1024
        size_mb = size_kb / 1024
        if size_mb >= 1:
            print(f"\n💾 Model size: {size_bytes:,} bytes ({size_mb:.2f} MB)")
        else:
            print(f"\n💾 Model size: {size_bytes:,} bytes ({size_kb:.2f} KB)")
    except:
        # Fallback to file size
        pass

def inspect_scaler(scaler):
    """Display information about the feature scaler"""
    print(f"\n{'='*70}")
    print(f"FEATURE SCALER (StandardScaler)")
    print(f"{'='*70}")
    
    print(f"\n📦 Scaler Type: {type(scaler).__name__}")
    
    if hasattr(scaler, 'n_features_in_'):
        print(f"📊 Number of features: {scaler.n_features_in_}")
    
    if hasattr(scaler, 'mean_'):
        print(f"\n📈 Feature Statistics (first 10 features):")
        print(f"\n   {'Feature':<20s} {'Mean':>12s} {'Std Dev':>12s}")
        print(f"   {'-'*46}")
        
        # Load feature names
        try:
            feature_names_path = Path("data/features/feature_names.txt")
            with open(feature_names_path, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
        except:
            feature_names = [f"Feature_{i}" for i in range(len(scaler.mean_))]
        
        for i in range(min(10, len(scaler.mean_))):
            feat_name = feature_names[i] if i < len(feature_names) else f"Feature_{i}"
            mean_val = scaler.mean_[i]
            scale_val = scaler.scale_[i] if hasattr(scaler, 'scale_') else 0
            print(f"   {feat_name:<20s} {mean_val:>12.6f} {scale_val:>12.6f}")
        
        if len(scaler.mean_) > 10:
            print(f"   ... and {len(scaler.mean_) - 10} more features")
    
    try:
        import pickle
        size_bytes = len(pickle.dumps(scaler))
        print(f"\n💾 Scaler size: {size_bytes:,} bytes ({size_bytes/1024:.2f} KB)")
    except:
        pass

def display_training_results():
    """Display training results from JSON"""
    results_path = Path("models/training_results.json")
    if not results_path.exists():
        print("\n⚠️  Training results JSON not found")
        return
    
    print(f"\n{'='*70}")
    print("📊 TRAINING RESULTS SUMMARY")
    print(f"{'='*70}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Validation Set Results
    print(f"\n{'VALIDATION SET PERFORMANCE':^70s}")
    print(f"\n{'Model':<22s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'AUC':>7s} {'Time':>9s}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        val_metrics = metrics.get('validation_metrics', {})
        name = model_name.replace('_', ' ').title()
        print(f"{name:<22s} "
              f"{val_metrics.get('accuracy', 0):>7.4f} "
              f"{val_metrics.get('precision', 0):>7.4f} "
              f"{val_metrics.get('recall', 0):>7.4f} "
              f"{val_metrics.get('f1_score', 0):>7.4f} "
              f"{val_metrics.get('auc_roc', 0):>7.4f} "
              f"{val_metrics.get('inference_time_ms', 0):>8.2f}ms")
    
    # Test Set Results
    print(f"\n{'TEST SET PERFORMANCE':^70s}")
    print(f"\n{'Model':<22s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'AUC':>7s} {'Time':>9s}")
    print("-" * 70)
    
    best_model = None
    best_f1 = 0
    
    for model_name, metrics in results.items():
        test_metrics = metrics.get('test_metrics', {})
        name = model_name.replace('_', ' ').title()
        f1 = test_metrics.get('f1_score', 0)
        
        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_model = name
            marker = " ⭐"
        
        print(f"{name:<22s} "
              f"{test_metrics.get('accuracy', 0):>7.4f} "
              f"{test_metrics.get('precision', 0):>7.4f} "
              f"{test_metrics.get('recall', 0):>7.4f} "
              f"{test_metrics.get('f1_score', 0):>7.4f} "
              f"{test_metrics.get('auc_roc', 0):>7.4f} "
              f"{test_metrics.get('inference_time_ms', 0):>8.2f}ms{marker}")
    
    print(f"\n✨ Best Model: {best_model} (F1-Score: {best_f1:.4f})")
    
    # Detailed test metrics for best model
    for model_name, metrics in results.items():
        name = model_name.replace('_', ' ').title()
        if name == best_model:
            test_metrics = metrics.get('test_metrics', {})
            print(f"\n📌 Detailed Metrics for {best_model}:")
            print(f"   • Accuracy: {test_metrics.get('accuracy', 0):.4f}")
            print(f"   • Precision: {test_metrics.get('precision', 0):.4f}")
            print(f"   • Recall: {test_metrics.get('recall', 0):.4f}")
            print(f"   • Sensitivity: {test_metrics.get('sensitivity', 0):.4f}")
            print(f"   • Specificity: {test_metrics.get('specificity', 0):.4f}")
            print(f"   • F1-Score: {test_metrics.get('f1_score', 0):.4f}")
            print(f"   • AUC-ROC: {test_metrics.get('auc_roc', 0):.4f}")
            print(f"\n   Confusion Matrix:")
            print(f"   ┌─────────────┬─────────────┐")
            print(f"   │  TN: {test_metrics.get('true_negatives', 0):>5d}  │  FP: {test_metrics.get('false_positives', 0):>5d}  │")
            print(f"   ├─────────────┼─────────────┤")
            print(f"   │  FN: {test_metrics.get('false_negatives', 0):>5d}  │  TP: {test_metrics.get('true_positives', 0):>5d}  │")
            print(f"   └─────────────┴─────────────┘")
            print(f"\n   Confidence Scores:")
            print(f"   • Mean: {test_metrics.get('mean_confidence', 0):.4f}")
            print(f"   • Median: {test_metrics.get('median_confidence', 0):.4f}")
            print(f"   • Range: [{test_metrics.get('min_confidence', 0):.4f}, {test_metrics.get('max_confidence', 0):.4f}]")

def main():
    """Main function to view all pickle files"""
    print("="*70)
    print("🔍 PKL MODEL VIEWER - RGB Anemia Detection".center(70))
    print("="*70)
    
    models_dir = Path("models")
    features_dir = Path("data/features")
    
    # List all pickle files
    print("\n📂 Found pickle files:")
    if models_dir.exists():
        for pkl_file in sorted(models_dir.glob("*.pkl")):
            size = pkl_file.stat().st_size
            size_kb = size / 1024
            size_mb = size_kb / 1024
            if size_mb >= 1:
                print(f"   • {pkl_file.name:<35s} ({size_mb:>6.2f} MB)")
            else:
                print(f"   • {pkl_file.name:<35s} ({size_kb:>6.2f} KB)")
    
    # Load and inspect models
    model_files = {
        "Random Forest": models_dir / "random_forest_model.pkl",
        "Gradient Boosting": models_dir / "gradient_boosting_model.pkl",
        "Logistic Regression": models_dir / "logistic_regression_model.pkl",
        "SVM": models_dir / "svm_model.pkl"
    }
    
    for model_name, model_path in model_files.items():
        if model_path.exists():
            print(f"\n\nLoading {model_name}...")
            model = load_model(model_path)
            if model is not None:
                inspect_model(model, model_name)
        else:
            print(f"\n⚠️  {model_name} not found at {model_path}")
    
    # Load and inspect scaler
    scaler_path = features_dir / "scaler.pkl"
    if scaler_path.exists():
        print(f"\n\nLoading Feature Scaler...")
        scaler = load_model(scaler_path)
        if scaler is not None:
            inspect_scaler(scaler)
    
    # Display training results
    display_training_results()
    
    print("\n" + "="*70)
    print("✅ Inspection complete!".center(70))
    print("="*70)

if __name__ == "__main__":
    main()
