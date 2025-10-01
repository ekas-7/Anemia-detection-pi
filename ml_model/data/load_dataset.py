"""
Dataset Loader for Eyes Defy Anemia Dataset
Downloads and loads the dataset from Kaggle using kagglehub
"""

import os
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import cv2
from tqdm import tqdm

class EyesDefyAnemiaDataset:
    """
    Handler for Eyes Defy Anemia dataset from Kaggle
    Dataset: harshwardhanfartale/eyes-defy-anemia
    """
    
    def __init__(self, data_dir='./data'):
        """
        Initialize dataset loader
        
        Args:
            data_dir: Directory to store downloaded and processed data
        """
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / 'raw'
        self.processed_data_dir = self.data_dir / 'processed'
        
        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_name = "harshwardhanfartale/eyes-defy-anemia"
        self.df = None
        self.images = []
        self.labels = []
        
    def download_dataset(self, file_path=""):
        """
        Download the Eyes Defy Anemia dataset from Kaggle
        
        Args:
            file_path: Specific file path within the dataset (empty for all files)
        
        Returns:
            DataFrame or dataset path depending on content type
        """
        print(f"Downloading dataset: {self.dataset_name}")
        print("This may take a few minutes on first run...")
        
        try:
            # Try loading as pandas dataframe first
            self.df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                self.dataset_name,
                file_path,
            )
            
            print(f"\n✓ Dataset loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            print(f"\nFirst 5 records:")
            print(self.df.head())
            print(f"\nColumn names: {list(self.df.columns)}")
            print(f"\nData types:")
            print(self.df.dtypes)
            
            # Save to CSV for future reference
            csv_path = self.raw_data_dir / 'dataset_info.csv'
            self.df.to_csv(csv_path, index=False)
            print(f"\n✓ Saved dataset info to: {csv_path}")
            
            return self.df
            
        except Exception as e:
            print(f"Could not load as pandas DataFrame: {e}")
            print("Attempting to download raw files...")
            
            # Download raw files using kagglehub
            dataset_path = kagglehub.dataset_download(self.dataset_name)
            print(f"\n✓ Dataset downloaded to: {dataset_path}")
            
            # Copy to our data directory
            if os.path.exists(dataset_path):
                print(f"Organizing files...")
                self._organize_downloaded_files(dataset_path)
            
            return dataset_path
    
    def _organize_downloaded_files(self, source_path):
        """
        Organize downloaded files into structured directories
        
        Args:
            source_path: Path where kagglehub downloaded the files
        """
        source = Path(source_path)
        
        # Look for common dataset structures
        for item in source.rglob('*'):
            if item.is_file():
                # Determine file type and destination
                if item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    # It's an image
                    dest_dir = self.raw_data_dir / 'images'
                    dest_dir.mkdir(exist_ok=True)
                    
                    # Try to determine label from parent folder name
                    parent_name = item.parent.name.lower()
                    if 'anemia' in parent_name or 'anem' in parent_name:
                        label_dir = dest_dir / 'anemia'
                    elif 'normal' in parent_name or 'healthy' in parent_name:
                        label_dir = dest_dir / 'normal'
                    else:
                        label_dir = dest_dir / 'unknown'
                    
                    label_dir.mkdir(exist_ok=True)
                    shutil.copy2(item, label_dir / item.name)
                    
                elif item.suffix.lower() in ['.csv', '.txt', '.json']:
                    # It's metadata
                    shutil.copy2(item, self.raw_data_dir / item.name)
        
        print(f"✓ Files organized in: {self.raw_data_dir}")
    
    def load_images_from_directory(self, target_size=(224, 224)):
        """
        Load images from organized directory structure
        
        Args:
            target_size: Resize images to this size (width, height)
        
        Returns:
            images: List of image arrays
            labels: List of corresponding labels
        """
        images_dir = self.raw_data_dir / 'images'
        
        if not images_dir.exists():
            print("No images directory found. Run download_dataset() first.")
            return None, None
        
        self.images = []
        self.labels = []
        
        # Label mapping
        label_map = {
            'anemia': 1,
            'normal': 0,
            'healthy': 0,
            'unknown': -1
        }
        
        print(f"\nLoading images from: {images_dir}")
        
        # Iterate through label directories
        for label_dir in images_dir.iterdir():
            if label_dir.is_dir():
                label_name = label_dir.name.lower()
                label_value = label_map.get(label_name, -1)
                
                print(f"\nProcessing {label_name} images...")
                
                # Load all images in this directory
                image_files = list(label_dir.glob('*'))
                
                for img_path in tqdm(image_files, desc=f"Loading {label_name}"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        try:
                            # Read image
                            img = cv2.imread(str(img_path))
                            if img is None:
                                continue
                            
                            # Convert BGR to RGB
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            
                            # Resize
                            img = cv2.resize(img, target_size)
                            
                            self.images.append(img)
                            self.labels.append(label_value)
                            
                        except Exception as e:
                            print(f"Error loading {img_path.name}: {e}")
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        print(f"\n✓ Loaded {len(self.images)} images")
        print(f"  Image shape: {self.images[0].shape if len(self.images) > 0 else 'N/A'}")
        print(f"  Label distribution: {np.bincount(self.labels[self.labels >= 0])}")
        
        return self.images, self.labels
    
    def get_dataset_statistics(self):
        """
        Print statistics about the loaded dataset
        """
        if self.df is not None:
            print("\n=== Dataset Statistics (DataFrame) ===")
            print(f"Total records: {len(self.df)}")
            print(f"\nColumn info:")
            print(self.df.describe())
            print(f"\nMissing values:")
            print(self.df.isnull().sum())
            
        if len(self.images) > 0:
            print("\n=== Image Dataset Statistics ===")
            print(f"Total images: {len(self.images)}")
            print(f"Image shape: {self.images[0].shape}")
            print(f"Data type: {self.images.dtype}")
            print(f"Value range: [{self.images.min()}, {self.images.max()}]")
            print(f"\nLabel distribution:")
            unique, counts = np.unique(self.labels, return_counts=True)
            for label, count in zip(unique, counts):
                label_name = 'Normal' if label == 0 else 'Anemia' if label == 1 else 'Unknown'
                print(f"  {label_name} ({label}): {count} ({count/len(self.labels)*100:.1f}%)")
    
    def save_processed_data(self, images, labels, split_name='train'):
        """
        Save processed images and labels to disk
        
        Args:
            images: Array of images
            labels: Array of labels
            split_name: Name of the split (train/val/test)
        """
        output_dir = self.processed_data_dir / split_name
        output_dir.mkdir(exist_ok=True)
        
        # Save as numpy arrays for fast loading
        np.save(output_dir / 'images.npy', images)
        np.save(output_dir / 'labels.npy', labels)
        
        print(f"✓ Saved {len(images)} images to: {output_dir}")


def main():
    """
    Main function to demonstrate dataset loading
    """
    print("="*60)
    print("Eyes Defy Anemia Dataset Loader")
    print("="*60)
    
    # Initialize dataset loader
    dataset = EyesDefyAnemiaDataset(data_dir='./data')
    
    # Download dataset (run once)
    print("\n[Step 1] Downloading dataset from Kaggle...")
    dataset.download_dataset(file_path="")
    
    # Load images
    print("\n[Step 2] Loading images...")
    images, labels = dataset.load_images_from_directory(target_size=(224, 224))
    
    # Show statistics
    print("\n[Step 3] Dataset Statistics")
    dataset.get_dataset_statistics()
    
    # Save processed data
    if images is not None and len(images) > 0:
        print("\n[Step 4] Saving processed data...")
        dataset.save_processed_data(images, labels, split_name='all')
    
    print("\n" + "="*60)
    print("✓ Dataset loading complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run preprocess.py to split and augment the data")
    print("2. Run extract_features.py to extract color features")
    print("3. Run train_ml_model.py to train the model")


if __name__ == "__main__":
    main()
