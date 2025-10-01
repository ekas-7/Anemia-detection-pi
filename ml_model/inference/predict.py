"""
Inference Script for ML Model
Lightweight inference with comprehensive output metrics
"""

import numpy as np
import cv2
from pathlib import Path
import time
import json
import joblib
import sys
import platform

# Add parent directory to path to import feature extractor
sys.path.append(str(Path(__file__).parent.parent))
from features.extract_features import FeatureExtractor


class AnemiaPredictor:
    """
    Lightweight anemia prediction from eye images
    """
    
    def __init__(self, model_path, scaler_path, model_type='random_forest'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler
            model_type: Type of model (for display)
        """
        self.model_type = model_type
        self.feature_extractor = FeatureExtractor()
        
        # Load model and scaler
        print(f"Loading {model_type} model...")
        self.model = joblib.load(model_path)
        self.feature_extractor.load_scaler(scaler_path)
        print("✓ Model and scaler loaded successfully")
        
        # Get system info
        self.os_info = self._get_system_info()
    
    def _get_system_info(self):
        """
        Get operating system and device information
        
        Returns:
            Dictionary with system info
        """
        return {
            'os': platform.system(),
            'os_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Load and preprocess image
        
        Args:
            image_path: Path to image file
            target_size: Target image size
        
        Returns:
            Preprocessed image
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def predict(self, image_path):
        """
        Predict anemia from eye image with comprehensive metrics
        
        Args:
            image_path: Path to eye image
        
        Returns:
            Dictionary with prediction results and metrics
        """
        # Start timing
        total_start_time = time.time()
        
        # Preprocess image
        preprocess_start = time.time()
        img = self.preprocess_image(image_path)
        preprocess_time = (time.time() - preprocess_start) * 1000  # ms
        
        # Extract features
        feature_start = time.time()
        features = self.feature_extractor.extract_all_features(img)
        features = features.reshape(1, -1)  # Reshape for single sample
        features_scaled = self.feature_extractor.transform_features(features)
        feature_time = (time.time() - feature_start) * 1000  # ms
        
        # Predict
        inference_start = time.time()
        prediction = self.model.predict(features_scaled)[0]
        
        # Get probability/confidence if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence_score = float(np.max(probabilities))
            class_probabilities = {
                'Normal': float(probabilities[0]),
                'Anemia': float(probabilities[1])
            }
        else:
            confidence_score = 1.0
            class_probabilities = None
        
        inference_time = (time.time() - inference_start) * 1000  # ms
        total_time = (time.time() - total_start_time) * 1000  # ms
        
        # Prepare result
        result = {
            'prediction': {
                'class': 'Anemia' if prediction == 1 else 'Normal',
                'class_id': int(prediction),
                'confidence_score': confidence_score,
                'class_probabilities': class_probabilities
            },
            'execution_info': {
                'model_type': self.model_type,
                'os': self.os_info['os'],
                'os_version': self.os_info['os_version'],
                'architecture': self.os_info['architecture'],
                'preprocessing_time_ms': round(preprocess_time, 2),
                'feature_extraction_time_ms': round(feature_time, 2),
                'inference_time_ms': round(inference_time, 2),
                'total_time_ms': round(total_time, 2)
            },
            'image_info': {
                'path': str(image_path),
                'size': img.shape[:2]
            }
        }
        
        return result
    
    def predict_batch(self, image_paths):
        """
        Predict on multiple images
        
        Args:
            image_paths: List of image paths
        
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        
        for img_path in image_paths:
            try:
                result = self.predict(img_path)
                results.append(result)
                print(f"✓ {Path(img_path).name}: {result['prediction']['class']} "
                      f"(confidence: {result['prediction']['confidence_score']:.2f})")
            except Exception as e:
                print(f"✗ Error processing {img_path}: {e}")
                results.append({'error': str(e), 'path': str(img_path)})
        
        return results
    
    def print_result(self, result):
        """
        Pretty print prediction result
        
        Args:
            result: Result dictionary
        """
        print("\n" + "="*60)
        print("ANEMIA DETECTION RESULT")
        print("="*60)
        
        if 'error' in result:
            print(f"\n✗ Error: {result['error']}")
            return
        
        pred = result['prediction']
        exec_info = result['execution_info']
        
        print(f"\n📊 PREDICTION")
        print(f"  Class:      {pred['class']}")
        print(f"  Confidence: {pred['confidence_score']:.4f} ({pred['confidence_score']*100:.2f}%)")
        
        if pred['class_probabilities']:
            print(f"\n  Class Probabilities:")
            for cls, prob in pred['class_probabilities'].items():
                print(f"    {cls:<10}: {prob:.4f} ({prob*100:.2f}%)")
        
        print(f"\n⚙️  EXECUTION INFO")
        print(f"  Model:      {exec_info['model_type']}")
        print(f"  OS:         {exec_info['os']} {exec_info['os_version'][:50]}")
        print(f"  Arch:       {exec_info['architecture']}")
        
        print(f"\n⏱️  TIMING BREAKDOWN")
        print(f"  Preprocessing:      {exec_info['preprocessing_time_ms']:>8.2f} ms")
        print(f"  Feature Extraction: {exec_info['feature_extraction_time_ms']:>8.2f} ms")
        print(f"  Inference:          {exec_info['inference_time_ms']:>8.2f} ms")
        print(f"  {'─'*40}")
        print(f"  TOTAL:              {exec_info['total_time_ms']:>8.2f} ms")
        
        print("\n" + "="*60)
    
    def save_result(self, result, output_path):
        """
        Save result to JSON file
        
        Args:
            result: Result dictionary
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✓ Result saved to: {output_path}")


def compare_models(image_path, models_dir='./models'):
    """
    Compare predictions from all available models
    
    Args:
        image_path: Path to test image
        models_dir: Directory containing trained models
    """
    models_dir = Path(models_dir)
    scaler_path = Path('./data/features/scaler.pkl')
    
    if not scaler_path.exists():
        print("Error: Scaler not found. Please train models first.")
        return
    
    # Find all model files
    model_files = list(models_dir.glob('*_model.pkl'))
    
    if len(model_files) == 0:
        print("No trained models found. Please run train_ml_model.py first.")
        return
    
    print("\n" + "="*60)
    print("COMPARING ALL MODELS")
    print("="*60)
    print(f"\nTest Image: {image_path}")
    print(f"Found {len(model_files)} trained models\n")
    
    results_comparison = []
    
    for model_file in model_files:
        model_name = model_file.stem.replace('_model', '')
        
        print(f"\n--- {model_name.replace('_', ' ').title()} ---")
        
        try:
            predictor = AnemiaPredictor(
                model_path=model_file,
                scaler_path=scaler_path,
                model_type=model_name
            )
            
            result = predictor.predict(image_path)
            
            print(f"  Prediction: {result['prediction']['class']}")
            print(f"  Confidence: {result['prediction']['confidence_score']:.4f}")
            print(f"  Total Time: {result['execution_info']['total_time_ms']:.2f} ms")
            
            results_comparison.append({
                'model': model_name,
                'result': result
            })
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Model':<25} {'Prediction':<15} {'Confidence':<12} {'Time (ms)':<10}")
    print("-" * 62)
    
    for item in results_comparison:
        model_name = item['model'].replace('_', ' ').title()
        pred = item['result']['prediction']
        exec_info = item['result']['execution_info']
        
        print(f"{model_name:<25} {pred['class']:<15} "
              f"{pred['confidence_score']:<12.4f} "
              f"{exec_info['total_time_ms']:<10.2f}")
    
    print("\n" + "="*60)


def main():
    """
    Main inference demo
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Anemia Detection Inference')
    parser.add_argument('--image', type=str, help='Path to eye image')
    parser.add_argument('--model', type=str, default='random_forest',
                       help='Model to use (random_forest, gradient_boosting, logistic_regression, svm)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all available models')
    parser.add_argument('--save', type=str, help='Path to save result JSON')
    
    args = parser.parse_args()
    
    # If no image provided, show usage
    if not args.image:
        print("="*60)
        print("Anemia Detection - Inference Demo")
        print("="*60)
        print("\nUsage:")
        print("  python predict.py --image <path_to_image>")
        print("  python predict.py --image <path_to_image> --model random_forest")
        print("  python predict.py --image <path_to_image> --compare")
        print("\nAvailable models:")
        print("  - random_forest (default)")
        print("  - gradient_boosting")
        print("  - logistic_regression")
        print("  - svm")
        return
    
    if args.compare:
        # Compare all models
        compare_models(args.image)
    else:
        # Single model prediction
        model_path = Path(f'./models/{args.model}_model.pkl')
        scaler_path = Path('./data/features/scaler.pkl')
        
        if not model_path.exists():
            print(f"Error: Model not found: {model_path}")
            print("Please run train_ml_model.py first or use --compare to see available models")
            return
        
        if not scaler_path.exists():
            print(f"Error: Scaler not found: {scaler_path}")
            print("Please run extract_features.py first")
            return
        
        # Initialize predictor
        predictor = AnemiaPredictor(
            model_path=model_path,
            scaler_path=scaler_path,
            model_type=args.model
        )
        
        # Predict
        result = predictor.predict(args.image)
        
        # Print result
        predictor.print_result(result)
        
        # Save if requested
        if args.save:
            predictor.save_result(result, args.save)


if __name__ == "__main__":
    main()
