"""
Feature Extraction for ML Model
Extracts color histograms, paleness index, and morphological features
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib


class FeatureExtractor:
    """
    Extract color-based and morphological features for anemia detection
    """
    
    def __init__(self):
        """Initialize feature extractor"""
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def extract_rgb_statistics(self, image):
        """
        Extract statistical features from RGB channels
        
        Args:
            image: RGB image (normalized or uint8)
        
        Returns:
            Dictionary of RGB features
        """
        # Ensure image is in [0, 255] range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        features = {}
        
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = image[:, :, i].flatten()
            
            features[f'{channel}_mean'] = np.mean(channel_data)
            features[f'{channel}_median'] = np.median(channel_data)
            features[f'{channel}_std'] = np.std(channel_data)
            features[f'{channel}_min'] = np.min(channel_data)
            features[f'{channel}_max'] = np.max(channel_data)
            features[f'{channel}_range'] = np.ptp(channel_data)
            features[f'{channel}_percentile_25'] = np.percentile(channel_data, 25)
            features[f'{channel}_percentile_75'] = np.percentile(channel_data, 75)
            features[f'{channel}_skewness'] = self._calculate_skewness(channel_data)
            features[f'{channel}_kurtosis'] = self._calculate_kurtosis(channel_data)
        
        # Inter-channel ratios
        features['RG_ratio'] = features['R_mean'] / (features['G_mean'] + 1e-6)
        features['RB_ratio'] = features['R_mean'] / (features['B_mean'] + 1e-6)
        features['GB_ratio'] = features['G_mean'] / (features['B_mean'] + 1e-6)
        
        return features
    
    def extract_hsv_features(self, image):
        """
        Extract features from HSV color space
        
        Args:
            image: RGB image
        
        Returns:
            Dictionary of HSV features
        """
        # Ensure image is uint8
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        features = {}
        
        for i, channel in enumerate(['H', 'S', 'V']):
            channel_data = hsv[:, :, i].flatten()
            
            features[f'{channel}_mean'] = np.mean(channel_data)
            features[f'{channel}_median'] = np.median(channel_data)
            features[f'{channel}_std'] = np.std(channel_data)
            features[f'{channel}_min'] = np.min(channel_data)
            features[f'{channel}_max'] = np.max(channel_data)
        
        return features
    
    def extract_lab_features(self, image):
        """
        Extract features from LAB color space
        
        Args:
            image: RGB image
        
        Returns:
            Dictionary of LAB features
        """
        # Ensure image is uint8
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        features = {}
        
        for i, channel in enumerate(['L', 'a', 'b']):
            channel_data = lab[:, :, i].flatten()
            
            features[f'LAB_{channel}_mean'] = np.mean(channel_data)
            features[f'LAB_{channel}_median'] = np.median(channel_data)
            features[f'LAB_{channel}_std'] = np.std(channel_data)
        
        return features
    
    def extract_color_histograms(self, image, bins=32):
        """
        Extract color histograms from RGB channels
        
        Args:
            image: RGB image
            bins: Number of histogram bins
        
        Returns:
            Dictionary with histogram features
        """
        # Ensure image is uint8
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        features = {}
        
        for i, channel in enumerate(['R', 'G', 'B']):
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            hist = hist.flatten() / (image.shape[0] * image.shape[1])  # Normalize
            
            for j, val in enumerate(hist):
                features[f'{channel}_hist_bin_{j}'] = val
        
        return features
    
    def calculate_paleness_index(self, image):
        """
        Calculate paleness index as an indicator of anemia
        Higher values indicate more paleness (potential anemia)
        
        Args:
            image: RGB image
        
        Returns:
            Dictionary with paleness features
        """
        # Ensure image is uint8
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Extract color channels
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        
        # Paleness index: higher red intensity is healthy, lower indicates anemia
        red_intensity = np.mean(r)
        
        # Redness ratio
        redness = red_intensity / (np.mean(g) + np.mean(b) + 1e-6)
        
        # Calculate overall brightness
        brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        
        # Paleness score (inverse of redness, normalized by brightness)
        paleness_score = (255 - red_intensity) / (brightness + 1e-6)
        
        features = {
            'red_intensity': red_intensity,
            'redness_ratio': redness,
            'brightness': brightness,
            'paleness_score': paleness_score,
        }
        
        return features
    
    def extract_texture_features(self, image):
        """
        Extract texture features using edge detection
        
        Args:
            image: RGB image
        
        Returns:
            Dictionary with texture features
        """
        # Convert to grayscale
        if image.max() <= 1.0:
            gray = (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        else:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Texture variance
        texture_variance = np.var(gray)
        
        # Local Binary Pattern (simplified)
        lbp_mean = np.mean(gray)
        lbp_std = np.std(gray)
        
        features = {
            'edge_density': edge_density,
            'texture_variance': texture_variance,
            'lbp_mean': lbp_mean,
            'lbp_std': lbp_std,
        }
        
        return features
    
    def extract_all_features(self, image):
        """
        Extract all features from an image
        
        Args:
            image: RGB image
        
        Returns:
            Feature vector as numpy array
        """
        all_features = {}
        
        # RGB statistics
        all_features.update(self.extract_rgb_statistics(image))
        
        # HSV features
        all_features.update(self.extract_hsv_features(image))
        
        # LAB features
        all_features.update(self.extract_lab_features(image))
        
        # Color histograms (using fewer bins to reduce dimensionality)
        all_features.update(self.extract_color_histograms(image, bins=16))
        
        # Paleness index
        all_features.update(self.calculate_paleness_index(image))
        
        # Texture features
        all_features.update(self.extract_texture_features(image))
        
        # Convert to array
        feature_vector = np.array(list(all_features.values()))
        
        # Store feature names on first call
        if len(self.feature_names) == 0:
            self.feature_names = list(all_features.keys())
        
        return feature_vector
    
    def extract_features_batch(self, images, labels=None):
        """
        Extract features from a batch of images
        
        Args:
            images: Array of images
            labels: Optional array of labels
        
        Returns:
            Feature matrix (n_samples, n_features)
            Labels (if provided)
        """
        print(f"Extracting features from {len(images)} images...")
        
        features_list = []
        
        for img in tqdm(images):
            features = self.extract_all_features(img)
            features_list.append(features)
        
        features_matrix = np.array(features_list)
        
        print(f"✓ Extracted {features_matrix.shape[1]} features per image")
        print(f"  Total feature matrix shape: {features_matrix.shape}")
        
        return features_matrix, labels
    
    def fit_scaler(self, features):
        """
        Fit StandardScaler on training features
        
        Args:
            features: Training feature matrix
        """
        self.scaler.fit(features)
        print("✓ Scaler fitted on training data")
    
    def transform_features(self, features):
        """
        Transform features using fitted scaler
        
        Args:
            features: Feature matrix
        
        Returns:
            Scaled feature matrix
        """
        return self.scaler.transform(features)
    
    def save_scaler(self, output_path):
        """
        Save the fitted scaler
        
        Args:
            output_path: Path to save the scaler
        """
        joblib.dump(self.scaler, output_path)
        print(f"✓ Scaler saved to: {output_path}")
    
    def load_scaler(self, scaler_path):
        """
        Load a saved scaler
        
        Args:
            scaler_path: Path to the saved scaler
        """
        self.scaler = joblib.load(scaler_path)
        print(f"✓ Scaler loaded from: {scaler_path}")
    
    @staticmethod
    def _calculate_skewness(data):
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    @staticmethod
    def _calculate_kurtosis(data):
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3


def process_and_save_features(data_dir='./data'):
    """
    Process all splits and save extracted features
    
    Args:
        data_dir: Data directory path
    """
    data_dir = Path(data_dir)
    processed_dir = data_dir / 'processed'
    features_dir = data_dir / 'features'
    features_dir.mkdir(exist_ok=True)
    
    extractor = FeatureExtractor()
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} set")
        print('='*60)
        
        # Load images and labels
        split_dir = processed_dir / split
        if not split_dir.exists():
            print(f"Warning: {split} directory not found, skipping...")
            continue
        
        images = np.load(split_dir / 'images.npy')
        labels = np.load(split_dir / 'labels.npy')
        
        print(f"Loaded {len(images)} images")
        
        # Extract features
        features, labels = extractor.extract_features_batch(images, labels)
        
        # Fit scaler on training data
        if split == 'train':
            extractor.fit_scaler(features)
            extractor.save_scaler(features_dir / 'scaler.pkl')
            
            # Save feature names
            feature_names_path = features_dir / 'feature_names.txt'
            with open(feature_names_path, 'w') as f:
                for name in extractor.feature_names:
                    f.write(f"{name}\n")
            print(f"✓ Saved feature names to: {feature_names_path}")
        
        # Scale features
        features_scaled = extractor.transform_features(features)
        
        # Save features
        output_dir = features_dir / split
        output_dir.mkdir(exist_ok=True)
        
        np.save(output_dir / 'features.npy', features_scaled)
        np.save(output_dir / 'labels.npy', labels)
        
        print(f"✓ Saved features to: {output_dir}")
        print(f"  Feature shape: {features_scaled.shape}")
        print(f"  Label distribution: {np.bincount(labels)}")


def main():
    """
    Main feature extraction pipeline
    """
    print("="*60)
    print("ML Model - Feature Extraction")
    print("="*60)
    
    # Extract and save features
    print("\n[Step 1] Extracting features from all splits...")
    process_and_save_features(data_dir='./data')
    
    print("\n" + "="*60)
    print("✓ Feature extraction complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run train_ml_model.py to train the model")
    print("2. Features are saved in ./data/features/")


if __name__ == "__main__":
    main()
