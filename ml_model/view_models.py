"""
View Model Details - Inspect saved .pkl files
============================================================
This script loads and displays information from the trained models
and other pickle files (scaler, etc.)
"""

import pickle
import numpy as np
import json
from pathlib import Path
import sys

def load_pickle_file(filepath):
    """Load a pickle file and return its contents"""
    try:
        with open(filepath, 'rb') as f:
            # Try different protocols and encodings
            return pickle.load(f, encoding='latin1')
    except Exception as e:
        try:
            # Try without encoding parameter
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e2:
            print(f"Error loading {filepath}: {e2}")
            return None

def inspect_model(model, model_name):
    """Display detailed information about a model"""
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")
    
    print(f"\nModel Type: {type(model).__name__}")
    print(f"Module: {type(model).__module__}")
    
    # Get model parameters
    if hasattr(model, 'get_params'):
        params = model.get_params()
        print(f"\nModel Parameters ({len(params)} total):")
        for key, value in sorted(params.items())[:15]:  # Show first 15
            print(f"  {key}: {value}")
        if len(params) > 15:
            print(f"  ... and {len(params) - 15} more parameters")
    
    # Model-specific information
    if hasattr(model, 'n_features_in_'):
        print(f"\nNumber of features: {model.n_features_in_}")
    
    if hasattr(model, 'classes_'):
        print(f"Classes: {model.classes_}")
    
    if hasattr(model, 'feature_importances_'):
        print(f"\nTop 10 Feature Importances:")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(min(10, len(importances))):
            idx = indices[i]
            print(f"  Feature {idx}: {importances[idx]:.6f}")
    
    if hasattr(model, 'n_estimators'):
        print(f"\nNumber of estimators: {model.n_estimators}")
    
    if hasattr(model, 'estimators_') and hasattr(model, 'n_estimators'):
        print(f"Number of trees/estimators trained: {len(model.estimators_)}")
    
    if hasattr(model, 'coef_'):
        print(f"\nModel coefficients shape: {model.coef_.shape}")
        print(f"Coefficients (first 10): {model.coef_[0][:10]}")
    
    if hasattr(model, 'intercept_'):
        print(f"Intercept: {model.intercept_}")
    
    # Memory size estimate
    import sys
    size_bytes = sys.getsizeof(pickle.dumps(model))
    size_mb = size_bytes / (1024 * 1024)
    print(f"\nModel size: {size_bytes:,} bytes ({size_mb:.2f} MB)")

def inspect_scaler(scaler):
    """Display information about the feature scaler"""
    print(f"\n{'='*60}")
    print(f"FEATURE SCALER")
    print(f"{'='*60}")
    
    print(f"\nScaler Type: {type(scaler).__name__}")
    
    if hasattr(scaler, 'n_features_in_'):
        print(f"Number of features: {scaler.n_features_in_}")
    
    if hasattr(scaler, 'mean_'):
        print(f"\nFeature means (first 10):")
        for i, mean in enumerate(scaler.mean_[:10]):
            print(f"  Feature {i}: {mean:.6f}")
    
    if hasattr(scaler, 'scale_'):
        print(f"\nFeature scales (first 10):")
        for i, scale in enumerate(scaler.scale_[:10]):
            print(f"  Feature {i}: {scale:.6f}")
    
    if hasattr(scaler, 'var_'):
        print(f"\nFeature variance (first 10):")
        for i, var in enumerate(scaler.var_[:10]):
            print(f"  Feature {i}: {var:.6f}")
    
    size_bytes = sys.getsizeof(pickle.dumps(scaler))
    print(f"\nScaler size: {size_bytes:,} bytes ({size_bytes/1024:.2f} KB)")

def main():
    """Main function to view all pickle files"""
    print("="*60)
    print("PICKLE FILE VIEWER - ML Model")
    print("="*60)
    
    models_dir = Path("models")
    features_dir = Path("data/features")
    
    # List all pickle files
    pkl_files = {
        "Models": list(models_dir.glob("*.pkl")) if models_dir.exists() else [],
        "Features": list(features_dir.glob("*.pkl")) if features_dir.exists() else []
    }
    
    print("\nFound pickle files:")
    for category, files in pkl_files.items():
        if files:
            print(f"\n{category}:")
            for f in files:
                size = f.stat().st_size
                print(f"  - {f.name} ({size:,} bytes, {size/1024:.2f} KB)")
    
    # Load and inspect models
    model_files = {
        "Random Forest": models_dir / "random_forest_model.pkl",
        "Gradient Boosting": models_dir / "gradient_boosting_model.pkl",
        "Logistic Regression": models_dir / "logistic_regression_model.pkl",
        "SVM": models_dir / "svm_model.pkl"
    }
    
    for model_name, model_path in model_files.items():
        if model_path.exists():
            model = load_pickle_file(model_path)
            if model is not None:
                inspect_model(model, model_name)
        else:
            print(f"\n⚠ {model_name} not found at {model_path}")
    
    # Load and inspect scaler
    scaler_path = features_dir / "scaler.pkl"
    if scaler_path.exists():
        scaler = load_pickle_file(scaler_path)
        if scaler is not None:
            inspect_scaler(scaler)
    else:
        print(f"\n⚠ Scaler not found at {scaler_path}")
    
    # Load feature names if available
    feature_names_path = features_dir / "feature_names.txt"
    if feature_names_path.exists():
        print(f"\n{'='*60}")
        print("FEATURE NAMES")
        print(f"{'='*60}")
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        print(f"\nTotal features: {len(feature_names)}")
        print("\nAll feature names:")
        for i, name in enumerate(feature_names):
            print(f"  {i+1:3d}. {name}")
    
    # Load training results if available
    results_path = models_dir / "training_results.json"
    if results_path.exists():
        print(f"\n{'='*60}")
        print("TRAINING RESULTS SUMMARY")
        print(f"{'='*60}")
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        print("\nTest Set Performance:")
        print(f"\n{'Model':<25} {'Accuracy':<12} {'F1-Score':<12} {'AUC-ROC':<12} {'Time (ms)':<12}")
        print("-" * 73)
        for model_name, metrics in results.items():
            test_metrics = metrics.get('test', {})
            print(f"{model_name:<25} "
                  f"{test_metrics.get('accuracy', 0):<12.4f} "
                  f"{test_metrics.get('f1_score', 0):<12.4f} "
                  f"{test_metrics.get('auc_roc', 0):<12.4f} "
                  f"{test_metrics.get('inference_time_ms', 0):<12.4f}")
    
    print("\n" + "="*60)
    print("✓ Inspection complete!")
    print("="*60)

if __name__ == "__main__":
    main()
