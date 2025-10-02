import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.data_loader import load_and_prepare_data
from src.model import build_unet, create_weighted_binary_crossentropy

# --- Constants ---
DATA_PATH = "../eyes-defy-anemia-dataset/" # Adjusted path assuming it's outside the pipeline folder

def main():
    """Main function to orchestrate the training process."""
    print("--- Step 1: Preparing Dataset ---")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset folder not found at '{DATA_PATH}'")
        print("Please place the 'eyes-defy-anemia-dataset' folder one level above the 'cnn_segmentation_pipeline' directory.")
        return
    
    print("--- Step 2: Loading Data and Building Model ---")
    X, Y = load_and_prepare_data(DATA_PATH)

    if X.size == 0:
        print("\nERROR: No training data was loaded.")
        return

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Calculate class weights to handle data imbalance
    neg = np.sum(1 - Y_train)
    pos = np.sum(Y_train)
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0) if neg > 0 else 0
    weight_for_1 = (1 / pos) * (total / 2.0) if pos > 0 else 0
    print(f"Weights calculated -> background (0): {weight_for_0:.2f}, conjunctiva (1): {weight_for_1:.2f}")

    # Create the custom loss function with these weights
    weighted_loss = create_weighted_binary_crossentropy(weight_for_0, weight_for_1)
    
    # Pass the custom loss function to the model builder
    model = build_unet(loss=weighted_loss)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
    ]

    print("\n--- Step 3: Training the Model ---")
    history = model.fit(
        X_train, Y_train, 
        validation_data=(X_val, Y_val), 
        batch_size=16, 
        epochs=50,
        callbacks=callbacks
    )
    
    print("\n--- Step 4: Saving Keras Model ---")
    model.save("unet_conjunctiva.keras")
    print("Model saved as unet_conjunctiva.keras")

    print("\n--- Step 5: Converting to TensorFlow Lite ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open("unet_conjunctiva.tflite", "wb") as f:
        f.write(tflite_model)
    print("Model converted and saved as unet_conjunctiva.tflite")

if __name__ == "__main__":
    main()
