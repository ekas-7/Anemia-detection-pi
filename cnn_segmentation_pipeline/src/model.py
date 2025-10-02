import tensorflow as tf
from .data_loader import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS

def create_weighted_binary_crossentropy(weight_for_0, weight_for_1):
    """
    Creates a weighted binary cross-entropy loss function.
    This is used to handle class imbalance by penalizing errors on the minority class (conjunctiva) more heavily.
    """
    def weighted_binary_crossentropy(y_true, y_pred):
        # Calculate element-wise binary crossentropy
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        
        # Apply the weights
        weights = y_true * weight_for_1 + (1. - y_true) * weight_for_0
        weighted_bce = weights * bce
        
        # Return the mean of the weighted loss
        return tf.keras.backend.mean(weighted_bce)
    
    # Set a name for the inner function, which Keras will save.
    weighted_binary_crossentropy.__name__ = 'weighted_binary_crossentropy'
    return weighted_binary_crossentropy


def build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), loss='binary_crossentropy'):
    """Builds the U-Net model architecture."""
    inputs = tf.keras.layers.Input(input_shape)

    # --- Contracting Path (Encoder) ---
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    # --- Bottleneck ---
    c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c5 = tf.keras.layers.Dropout(0.2)(c5)
    c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # --- Expanding Path (Decoder) ---
    u7 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
    u7 = tf.keras.layers.concatenate([u7, c2])
    c7 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.1)(c7)
    c7 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c1])
    c8 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c8)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    # Compile the model with the provided loss function
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    
    return model
