import numpy as np
import pickle
from PIL import Image
import tensorflow as tf
from pathlib import Path

def load_model():
    """
    Load the trained pneumonia detection model
    """
    # Define paths to different model formats
    utils_dir = Path(__file__).parent
    keras_model_path = utils_dir / "pneumo_cnn.keras"
    h5_model_path = utils_dir / "pneumo_cnn.h5"
    
    print(f"Looking for model files in: {utils_dir}")
    
    # Try loading the .keras file first
    if keras_model_path.exists():
        try:
            # Try with compile=False first (sometimes helps with compatibility)
            model = tf.keras.models.load_model(str(keras_model_path), compile=False)
            print(f"✓ Successfully loaded Keras model from {keras_model_path}")
            
            # Manually compile the model
            model.compile(
                optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Build the model by calling it with a dummy input
            # This ensures all layers have defined outputs for Grad-CAM
            dummy_input = np.zeros((1, 150, 150, 1), dtype=np.float32)
            _ = model.predict(dummy_input, verbose=0)
            print("✓ Model built and ready for Grad-CAM")
            
            return model
        except Exception as e:
            print(f"Failed to load .keras file: {e}")
            print("Trying alternative loading method...")
    
    # Try .h5 format as fallback
    if h5_model_path.exists():
        try:
            model = tf.keras.models.load_model(str(h5_model_path), compile=False)
            print(f"✓ Successfully loaded H5 model from {h5_model_path}")
            
            model.compile(
                optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Build the model by calling it with a dummy input
            dummy_input = np.zeros((1, 150, 150, 1), dtype=np.float32)
            _ = model.predict(dummy_input, verbose=0)
            print("✓ Model built and ready for Grad-CAM")
            
            return model
        except Exception as e:
            print(f"Failed to load .h5 file: {e}")
    
    # If both fail, build the architecture and try to load weights
    print("Building model architecture from scratch...")
    model = build_model_architecture()
    
    # Try to find weights file
    weights_path = utils_dir / "model_weights.h5"
    if weights_path.exists():
        try:
            model.load_weights(str(weights_path))
            print(f"✓ Loaded weights from {weights_path}")
            return model
        except Exception as e:
            print(f"Failed to load weights: {e}")
    
    print("⚠️ Warning: Using untrained model. Predictions will not be accurate!")
    return model

def build_model_architecture():
    """
    Build the CNN model architecture matching the training
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=(150, 150, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'),
        tf.keras.layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_image(image):
    """
    Preprocess the image for model prediction
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed numpy array
    """
    # Resize to model input size
    img = image.resize((150, 150))
    
    # Convert to grayscale (model expects 1 channel, not 3)
    if img.mode != 'L':
        img = img.convert('L')
    
    # Convert to array
    img_array = np.array(img)
    
    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0
    
    # Add channel dimension and batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    
    return img_array

def predict(model, processed_image):
    """
    Make prediction on the preprocessed image
    
    Args:
        model: Trained model
        processed_image: Preprocessed image array
        
    Returns:
        tuple: (prediction_label, confidence)
    """
    # Make prediction
    prediction = model.predict(processed_image, verbose=0)
    
    # Get the raw prediction value (probability for class 1)
    raw_prediction = float(prediction[0][0])
    
    # Based on training: labels = ['PNEUMONIA', 'NORMAL']
    # where class 0 = PNEUMONIA, class 1 = NORMAL
    # Model outputs sigmoid probability for class 1 (NORMAL)
    
    if raw_prediction > 0.5:
        # High probability means NORMAL (class 1)
        label = 'NORMAL'
        confidence = raw_prediction
    else:
        # Low probability means PNEUMONIA (class 0)
        label = 'PNEUMONIA'
        confidence = 1 - raw_prediction
    
    return label, confidence

def get_model_summary():
    """
    Get a summary of the model architecture
    
    Returns:
        dict: Model information
    """
    return {
        'architecture': 'Convolutional Neural Network (CNN)',
        'input_shape': '150x150x1 (Grayscale)',
        'layers': [
            'Conv2D (32 filters) + BatchNorm + MaxPooling',
            'Conv2D (64 filters) + Dropout(0.1) + BatchNorm + MaxPooling',
            'Conv2D (64 filters) + BatchNorm + MaxPooling',
            'Conv2D (128 filters) + Dropout(0.2) + BatchNorm + MaxPooling',
            'Conv2D (256 filters) + Dropout(0.2) + BatchNorm + MaxPooling',
            'Flatten',
            'Dense (128 units) + Dropout(0.2)',
            'Dense (1 unit, sigmoid)'
        ],
        'total_params': '~2.3M',
        'optimizer': 'RMSprop',
        'loss': 'Binary Crossentropy',
        'expected_accuracy': '~92.6%'
    }
