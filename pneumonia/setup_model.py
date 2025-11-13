"""
Model Setup Script

This script helps set up the pneumonia detection model.
If you have a pre-trained model, place it in the utils folder as 'model.pkl'.
Otherwise, this script will create the model architecture.
"""

import tensorflow as tf
import pickle
from pathlib import Path

def create_model():
    """
    Create the CNN model architecture based on the Kaggle notebook
    """
    model = tf.keras.models.Sequential([
        # First Convolutional Block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Second Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Third Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Fourth Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Flatten and Dense Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        
        # Output Layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("Pneumonia Detection Model Setup")
    print("=" * 50)
    
    # Check if model already exists
    model_path = Path(__file__).parent / "utils" / "model.pkl"
    
    if model_path.exists():
        print(f"\nModel file found at: {model_path}")
        print("No setup needed!")
    else:
        print("\nNo model file found. Creating model architecture...")
        model = create_model()
        
        print("\nModel Summary:")
        model.summary()
        
        print("\n" + "=" * 50)
        print("IMPORTANT: This is just the model architecture.")
        print("To use the model for predictions, you need to:")
        print("1. Train the model on chest X-ray data, OR")
        print("2. Load pre-trained weights")
        print("\nIf you have pre-trained weights (.h5 file),")
        print("place them in the utils folder as 'model_weights.h5'")
        print("=" * 50)

if __name__ == "__main__":
    main()
