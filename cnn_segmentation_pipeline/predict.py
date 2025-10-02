import os
import cv2
import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from src.data_loader import IMG_HEIGHT, IMG_WIDTH
from src.model import create_weighted_binary_crossentropy

def predict_and_show(image_path, model_path, tflite_path):
    """Loads models and predicts on a single image."""
    print(f"--- Running Prediction on {image_path} ---")
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"\nERROR: Could not read image file at '{image_path}'.")
        return

    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    input_data = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)

    # Load the Keras model with the custom loss function object
    keras_model = tf.keras.models.load_model(
        model_path, 
        custom_objects={'weighted_binary_crossentropy': create_weighted_binary_crossentropy(1.0, 1.0)}
    )
    keras_pred_mask = (keras_model.predict(input_data)[0] > 0.5).astype(np.uint8)

    # Load the TFLite model and predict
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_pred_mask = (interpreter.get_tensor(output_details[0]['index'])[0] > 0.5).astype(np.uint8)

    # Apply masks to the original image to show color
    keras_result = cv2.bitwise_and(img_rgb, img_rgb, mask=keras_pred_mask)
    tflite_result = cv2.bitwise_and(img_rgb, img_rgb, mask=tflite_pred_mask)

    # --- Visualization ---
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Keras Prediction (Color)")
    plt.imshow(keras_result)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("TFLite Prediction (Color)")
    plt.imshow(tflite_result)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction script for conjunctiva segmentation.")
    parser.add_argument("--image", type=str, required=True, help="Path to an image file for prediction.")
    parser.add_argument("--model", type=str, default="unet_conjunctiva.keras", help="Path to the trained Keras model file.")
    parser.add_argument("--tflite", type=str, default="unet_conjunctiva.tflite", help="Path to the trained TFLite model file.")
    args = parser.parse_args()

    if not os.path.exists(args.model) or not os.path.exists(args.tflite):
        print(f"Error: Model files not found. Please run the training script first to generate '{args.model}' and '{args.tflite}'.")
    else:
        predict_and_show(args.image, args.model, args.tflite)
