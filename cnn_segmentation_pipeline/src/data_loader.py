import os
import cv2
import numpy as np
import glob

# --- Constants ---
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

def load_and_prepare_data(data_path):
    """
    Loads images and masks from the dataset folder, searching recursively.
    It pairs images with their corresponding masks, combining multiple masks for a single image.
    
    Args:
        data_path (str): The path to the root dataset folder.

    Returns:
        A tuple of (X, Y) numpy arrays, where X contains images and Y contains masks.
    """
    print("Searching for image and mask pairs based on naming convention (e.g., 'img.jpg' and 'img_palpebral.png')...")
    
    all_files = glob.glob(os.path.join(data_path, '**', '*.*'), recursive=True)

    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg'))]
    mask_files = [f for f in all_files if f.lower().endswith('.png')]

    image_to_masks_map = {}
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        image_to_masks_map[base_name] = []

    for mask_path in mask_files:
        mask_basename = os.path.basename(mask_path)
        for image_base_name in image_to_masks_map.keys():
            if mask_basename.startswith(image_base_name):
                image_to_masks_map[image_base_name].append(mask_path)
                break
    
    X_paths, Y_mask_groups = [], []
    for base_name, masks in image_to_masks_map.items():
        if masks:
            img_path = next((f for f in image_files if os.path.splitext(os.path.basename(f))[0] == base_name), None)
            if img_path:
                X_paths.append(img_path)
                Y_mask_groups.append(masks)

    print(f"Found {len(X_paths)} images with corresponding combined masks.")

    if not X_paths:
        return np.array([]), np.array([])

    X = np.zeros((len(X_paths), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    Y = np.zeros((len(X_paths), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

    print("Loading, combining, and resizing image/mask pairs...")
    for i, (img_path, mask_group) in enumerate(zip(X_paths, Y_mask_groups)):
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        X[i] = img / 255.0

        combined_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        for mask_path in mask_group:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask_resized = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
                combined_mask = cv2.bitwise_or(combined_mask, mask_resized)
        
        Y[i] = (np.expand_dims(combined_mask, axis=-1) > 0).astype(np.float32)

    return X, Y
