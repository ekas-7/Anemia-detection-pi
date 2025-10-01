"""
RGB Intensity Inference - Test Individual Images
============================================================
Load trained RGB intensity model and test on new images
"""

import cv2
import numpy as np
from pathlib import Path
import json
import sys
import argparse

# Import the analyzer from training script
sys.path.append(str(Path(__file__).parent))
from train_rgb_intensity import RGBIntensityAnalyzer


def test_single_image(image_path, model_path='models/rgb_intensity_model.json'):
    """Test RGB intensity model on a single image"""
    
    print("="*70)
    print("RGB INTENSITY ANEMIA DETECTION - INFERENCE")
    print("="*70)
    
    # Load model
    analyzer = RGBIntensityAnalyzer()
    analyzer.load_model(model_path)
    
    # Load and process image
    print(f"\nLoading image: {image_path}")
    image = analyzer.load_image(image_path)
    
    # Extract features
    print("Extracting RGB intensity features...")
    features = analyzer.extract_rgb_intensity_features(image)
    
    # Make prediction
    result = analyzer.predict_from_features(features)
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    print(f"\n🔍 Prediction: {result['prediction_label']}")
    print(f"📊 Confidence: {result['confidence']:.2%}")
    print(f"📈 Anemia Score: {result['anemia_score']:.2f} / 4.0")
    
    print(f"\n📋 Key RGB Intensity Features:")
    print(f"  Red Mean Intensity:    {features['red_mean']:.2f}")
    print(f"  Redness Ratio (R/G):   {features['redness_ratio']:.3f}")
    print(f"  Paleness Score:        {features['paleness_score']:.3f}")
    print(f"  Red Dominance:         {features['red_dominance']:.2f}")
    print(f"  Overall Brightness:    {features['brightness']:.2f}")
    
    print(f"\n🎯 Model Thresholds:")
    print(f"  Red Intensity:         {analyzer.thresholds['red_intensity']:.2f}")
    print(f"  Redness Ratio:         {analyzer.thresholds['redness_ratio']:.3f}")
    print(f"  Paleness Score:        {analyzer.thresholds['paleness_score']:.3f}")
    
    if result['confidence_factors']:
        print(f"\n⚠️  Anemia Indicators Detected:")
        for factor, strength in result['confidence_factors']:
            print(f"  • {factor}: {strength:.2%} deviation")
    
    print("\n" + "="*70)
    
    return result


def test_batch_images(image_dir, model_path='models/rgb_intensity_model.json'):
    """Test RGB intensity model on multiple images"""
    
    image_dir = Path(image_dir)
    
    print("="*70)
    print("RGB INTENSITY BATCH TESTING")
    print("="*70)
    
    # Load model
    analyzer = RGBIntensityAnalyzer()
    analyzer.load_model(model_path)
    
    # Find all images
    image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
    
    print(f"\nFound {len(image_files)} images")
    
    results = []
    
    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        
        try:
            image = analyzer.load_image(img_path)
            features = analyzer.extract_rgb_intensity_features(image)
            result = analyzer.predict_from_features(features)
            
            results.append({
                'image': img_path.name,
                'prediction': result['prediction_label'],
                'confidence': result['confidence'],
                'red_intensity': features['red_mean'],
                'redness_ratio': features['redness_ratio'],
                'paleness_score': features['paleness_score']
            })
            
            print(f"  → {result['prediction_label']} (confidence: {result['confidence']:.2%})")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("BATCH RESULTS SUMMARY")
    print("="*70)
    
    normal_count = sum(1 for r in results if r['prediction'] == 'Normal')
    anemia_count = sum(1 for r in results if r['prediction'] == 'Anemia')
    
    print(f"\nTotal images: {len(results)}")
    print(f"Normal:  {normal_count} ({normal_count/len(results)*100:.1f}%)")
    print(f"Anemia:  {anemia_count} ({anemia_count/len(results)*100:.1f}%)")
    
    avg_confidence = np.mean([r['confidence'] for r in results])
    print(f"\nAverage confidence: {avg_confidence:.2%}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='RGB Intensity Anemia Detection Inference')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--batch', type=str, help='Path to directory of images')
    parser.add_argument('--model', type=str, default='models/rgb_intensity_model.json',
                       help='Path to model file')
    
    args = parser.parse_args()
    
    if args.image:
        test_single_image(args.image, args.model)
    elif args.batch:
        test_batch_images(args.batch, args.model)
    else:
        print("Please specify --image or --batch")
        print("\nExamples:")
        print("  python test_rgb_intensity.py --image path/to/image.png")
        print("  python test_rgb_intensity.py --batch path/to/images/folder")


if __name__ == "__main__":
    main()
