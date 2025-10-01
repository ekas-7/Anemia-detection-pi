"""
Image Preprocessing and Augmentation for ML Model
Handles train/val/test splitting, normalization, and augmentation
"""

import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt


class ImagePreprocessor:
    """
    Preprocessing pipeline for anemia detection images
    """
    
    def __init__(self, target_size=(224, 224), data_dir='./data'):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size (width, height)
            data_dir: Data directory path
        """
        self.target_size = target_size
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed'
        self.processed_dir.mkdir(exist_ok=True)
        
    def normalize_image(self, image, method='standard'):
        """
        Normalize image pixels
        
        Args:
            image: Input image (uint8)
            method: Normalization method ('standard', 'minmax', 'imagenet')
        
        Returns:
            Normalized image
        """
        image = image.astype(np.float32)
        
        if method == 'standard':
            # Standardize to [0, 1]
            image = image / 255.0
            
        elif method == 'minmax':
            # Min-max normalization
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
        elif method == 'imagenet':
            # ImageNet normalization
            image = image / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
            
        return image
    
    def enhance_contrast(self, image):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Input RGB image
        
        Returns:
            Contrast enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def denoise_image(self, image):
        """
        Apply denoising
        
        Args:
            image: Input image
        
        Returns:
            Denoised image
        """
        # Bilateral filter preserves edges while removing noise
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        return denoised
    
    def get_augmentation_pipeline(self, is_training=True):
        """
        Get augmentation pipeline using albumentations
        
        Args:
            is_training: If True, apply training augmentations
        
        Returns:
            Albumentations compose object
        """
        if is_training:
            transform = A.Compose([
                # Geometric transformations
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                
                # Color transformations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.5
                ),
                A.RGBShift(
                    r_shift_limit=10,
                    g_shift_limit=10,
                    b_shift_limit=10,
                    p=0.3
                ),
                
                # Noise and blur
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ], p=0.3),
                
                # Quality degradation (simulate real-world conditions)
                A.OneOf([
                    A.ImageCompression(quality_lower=80, quality_upper=100, p=1.0),
                    A.Downscale(scale_min=0.75, scale_max=0.95, p=1.0),
                ], p=0.2),
            ])
        else:
            # Validation/test: no augmentation
            transform = A.Compose([])
        
        return transform
    
    def preprocess_single_image(self, image, enhance=True, denoise=True, normalize=True):
        """
        Preprocess a single image
        
        Args:
            image: Input image (RGB)
            enhance: Apply contrast enhancement
            denoise: Apply denoising
            normalize: Apply normalization
        
        Returns:
            Preprocessed image
        """
        # Resize
        img = cv2.resize(image, self.target_size)
        
        # Enhance contrast
        if enhance:
            img = self.enhance_contrast(img)
        
        # Denoise
        if denoise:
            img = self.denoise_image(img)
        
        # Normalize
        if normalize:
            img = self.normalize_image(img, method='standard')
        
        return img
    
    def split_dataset(self, images, labels, test_size=0.15, val_size=0.15, random_state=42):
        """
        Split dataset into train/val/test
        
        Args:
            images: Array of images
            labels: Array of labels
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
        
        Returns:
            Dictionary with train/val/test splits
        """
        # Check if we have valid labels
        unique_labels = np.unique(labels)
        print(f"\nUnique labels in dataset: {unique_labels}")
        
        # If all labels are -1 (unknown), assign random labels for demonstration
        if len(unique_labels) == 1 and unique_labels[0] == -1:
            print("\nWarning: All images have unknown labels!")
            print("Assigning random labels for demonstration purposes...")
            print("In production, you should have properly labeled data.")
            # Randomly assign 50% as anemia (1) and 50% as normal (0)
            np.random.seed(random_state)
            labels = np.random.randint(0, 2, size=len(labels))
            print(f"Assigned labels - Anemia: {np.sum(labels == 1)}, Normal: {np.sum(labels == 0)}")
        else:
            # Remove unknown labels (-1) if we have other valid labels
            valid_mask = labels >= 0
            if np.sum(~valid_mask) > 0:
                print(f"\nRemoving {np.sum(~valid_mask)} samples with unknown labels")
                images = images[valid_mask]
                labels = labels[valid_mask]
        
        print(f"Total samples for training: {len(images)}")
        
        # First split: train+val vs test
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                images, labels,
                test_size=test_size,
                random_state=random_state,
                stratify=labels
            )
        except ValueError as e:
            print(f"Warning: Stratification failed ({e}). Using non-stratified split.")
            X_temp, X_test, y_temp, y_test = train_test_split(
                images, labels,
                test_size=test_size,
                random_state=random_state
            )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=y_temp
            )
        except ValueError as e:
            print(f"Warning: Stratification failed ({e}). Using non-stratified split.")
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=random_state
            )
        
        print(f"\nDataset split:")
        print(f"  Train: {len(X_train)} samples ({len(X_train)/len(images)*100:.1f}%)")
        print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(images)*100:.1f}%)")
        print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(images)*100:.1f}%)")
        
        return {
            'train': {'images': X_train, 'labels': y_train},
            'val': {'images': X_val, 'labels': y_val},
            'test': {'images': X_test, 'labels': y_test}
        }
    
    def augment_dataset(self, images, labels, augmentation_factor=3):
        """
        Augment dataset by creating multiple versions of each image
        
        Args:
            images: Array of images
            labels: Array of labels
            augmentation_factor: Number of augmented versions per image
        
        Returns:
            Augmented images and labels
        """
        aug_pipeline = self.get_augmentation_pipeline(is_training=True)
        
        augmented_images = []
        augmented_labels = []
        
        print(f"Augmenting dataset (factor: {augmentation_factor})...")
        
        for img, label in tqdm(zip(images, labels), total=len(images)):
            # Add original
            augmented_images.append(img)
            augmented_labels.append(label)
            
            # Add augmented versions
            for _ in range(augmentation_factor - 1):
                # Convert to uint8 for albumentations
                img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                
                # Apply augmentation
                augmented = aug_pipeline(image=img_uint8)['image']
                
                # Convert back to float if needed
                if img.max() <= 1.0:
                    augmented = augmented.astype(np.float32) / 255.0
                
                augmented_images.append(augmented)
                augmented_labels.append(label)
        
        return np.array(augmented_images), np.array(augmented_labels)
    
    def process_and_save_splits(self, images, labels, augment_train=True):
        """
        Process and save all data splits
        
        Args:
            images: Array of images
            labels: Array of labels
            augment_train: Whether to augment training data
        """
        print("\n" + "="*60)
        print("Processing Dataset Splits")
        print("="*60)
        
        # Split dataset
        splits = self.split_dataset(images, labels)
        
        # Process each split
        for split_name, data in splits.items():
            print(f"\n[Processing {split_name.upper()} set]")
            
            images_split = data['images']
            labels_split = data['labels']
            
            # Preprocess images
            print("Preprocessing images...")
            processed_images = []
            for img in tqdm(images_split):
                # Apply preprocessing
                processed = self.preprocess_single_image(img)
                processed_images.append(processed)
            
            processed_images = np.array(processed_images)
            
            # Augment training data
            if split_name == 'train' and augment_train:
                print("Augmenting training data...")
                processed_images, labels_split = self.augment_dataset(
                    processed_images,
                    labels_split,
                    augmentation_factor=3
                )
            
            # Save
            output_dir = self.processed_dir / split_name
            output_dir.mkdir(exist_ok=True)
            
            np.save(output_dir / 'images.npy', processed_images)
            np.save(output_dir / 'labels.npy', labels_split)
            
            print(f"✓ Saved {len(processed_images)} samples to {output_dir}")
            print(f"  Image shape: {processed_images.shape}")
            print(f"  Label distribution: {np.bincount(labels_split)}")
    
    def visualize_samples(self, images, labels, num_samples=9):
        """
        Visualize sample images
        
        Args:
            images: Array of images
            labels: Array of labels
            num_samples: Number of samples to show
        """
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        indices = np.random.choice(len(images), size=min(num_samples, len(images)), replace=False)
        
        for i, idx in enumerate(indices):
            img = images[idx]
            label = labels[idx]
            
            # Denormalize if needed
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            axes[i].imshow(img)
            axes[i].set_title(f"Label: {'Anemia' if label == 1 else 'Normal'}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.processed_dir / 'sample_visualization.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {self.processed_dir / 'sample_visualization.png'}")
        plt.close()


def main():
    """
    Main preprocessing pipeline
    """
    print("="*60)
    print("ML Model - Data Preprocessing")
    print("="*60)
    
    # Load raw data
    data_dir = Path('./data')
    processed_dir = data_dir / 'processed'
    
    # Check if raw data exists
    all_data_path = processed_dir / 'all'
    if not all_data_path.exists():
        print("\nError: Raw data not found!")
        print("Please run load_dataset.py first to download the dataset.")
        return
    
    print("\n[Step 1] Loading raw data...")
    images = np.load(all_data_path / 'images.npy')
    labels = np.load(all_data_path / 'labels.npy')
    print(f"✓ Loaded {len(images)} images")
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=(224, 224), data_dir=data_dir)
    
    # Process and save splits
    print("\n[Step 2] Processing and splitting dataset...")
    preprocessor.process_and_save_splits(images, labels, augment_train=True)
    
    # Visualize samples
    print("\n[Step 3] Creating visualizations...")
    train_images = np.load(processed_dir / 'train' / 'images.npy')
    train_labels = np.load(processed_dir / 'train' / 'labels.npy')
    preprocessor.visualize_samples(train_images, train_labels)
    
    print("\n" + "="*60)
    print("✓ Preprocessing complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run extract_features.py to extract color features")
    print("2. Run train_ml_model.py to train the model")


if __name__ == "__main__":
    main()
