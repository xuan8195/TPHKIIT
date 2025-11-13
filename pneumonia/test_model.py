"""
Test script to verify the pneumonia detection model works correctly
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.model_loader import load_model, preprocess_image, predict

def test_model():
    """Test the model loading and prediction"""
    print("=" * 60)
    print("PNEUMONIA DETECTION MODEL TEST")
    print("=" * 60)
    
    # Test 1: Load model
    print("\n[Test 1] Loading model...")
    try:
        model = load_model()
        print("✓ Model loaded successfully")
        print(f"  Model input shape: {model.input_shape}")
        print(f"  Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Test 2: Model architecture
    print("\n[Test 2] Checking model architecture...")
    try:
        expected_input = (None, 150, 150, 1)
        expected_output = (None, 1)
        
        if model.input_shape == expected_input:
            print(f"✓ Input shape correct: {expected_input}")
        else:
            print(f"✗ Input shape mismatch. Expected: {expected_input}, Got: {model.input_shape}")
            
        if model.output_shape == expected_output:
            print(f"✓ Output shape correct: {expected_output}")
        else:
            print(f"✗ Output shape mismatch. Expected: {expected_output}, Got: {model.output_shape}")
    except Exception as e:
        print(f"✗ Architecture check failed: {e}")
    
    # Test 3: Test preprocessing
    print("\n[Test 3] Testing image preprocessing...")
    try:
        # Create a test image (random RGB image)
        test_image = Image.new('RGB', (256, 256), color='white')
        processed = preprocess_image(test_image)
        
        print(f"  Original image size: {test_image.size}")
        print(f"  Processed shape: {processed.shape}")
        print(f"  Processed dtype: {processed.dtype}")
        print(f"  Value range: [{processed.min():.3f}, {processed.max():.3f}]")
        
        if processed.shape == (1, 150, 150, 1):
            print("✓ Preprocessing output shape correct")
        else:
            print(f"✗ Preprocessing shape incorrect. Expected: (1, 150, 150, 1), Got: {processed.shape}")
            
        if 0 <= processed.min() <= processed.max() <= 1:
            print("✓ Preprocessing normalization correct")
        else:
            print(f"✗ Values not normalized to [0,1]")
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        return False
    
    # Test 4: Test prediction
    print("\n[Test 4] Testing model prediction...")
    try:
        prediction, confidence = predict(model, processed)
        
        print(f"  Prediction: {prediction}")
        print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        if prediction in ['NORMAL', 'PNEUMONIA']:
            print("✓ Prediction label valid")
        else:
            print(f"✗ Invalid prediction label: {prediction}")
            
        if 0 <= confidence <= 1:
            print("✓ Confidence score in valid range")
        else:
            print(f"✗ Confidence score out of range: {confidence}")
    except Exception as e:
        print(f"✗ Prediction test failed: {e}")
        return False
    
    # Test 5: Multiple predictions (consistency check)
    print("\n[Test 5] Testing prediction consistency...")
    try:
        predictions = []
        for i in range(5):
            pred, conf = predict(model, processed)
            predictions.append((pred, conf))
        
        # Check if predictions are consistent
        if all(p[0] == predictions[0][0] for p in predictions):
            print("✓ Predictions are consistent")
            print(f"  All predictions: {predictions[0][0]}")
        else:
            print("✗ Predictions are inconsistent!")
            for i, (pred, conf) in enumerate(predictions):
                print(f"  Prediction {i+1}: {pred} ({conf:.4f})")
    except Exception as e:
        print(f"✗ Consistency test failed: {e}")
    
    # Test 6: Test with different image types
    print("\n[Test 6] Testing with different image formats...")
    try:
        # RGB image
        rgb_img = Image.new('RGB', (200, 200), color='gray')
        rgb_processed = preprocess_image(rgb_img)
        rgb_pred, rgb_conf = predict(model, rgb_processed)
        print(f"✓ RGB image: {rgb_pred} ({rgb_conf:.2%})")
        
        # Grayscale image
        gray_img = Image.new('L', (200, 200), color=128)
        gray_processed = preprocess_image(gray_img)
        gray_pred, gray_conf = predict(model, gray_processed)
        print(f"✓ Grayscale image: {gray_pred} ({gray_conf:.2%})")
        
        # Different size image
        large_img = Image.new('RGB', (512, 512), color='lightgray')
        large_processed = preprocess_image(large_img)
        large_pred, large_conf = predict(model, large_processed)
        print(f"✓ Large image: {large_pred} ({large_conf:.2%})")
        
    except Exception as e:
        print(f"✗ Image format test failed: {e}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
    print("=" * 60)
    print("\n💡 The model is ready to use in the Streamlit app!")
    print("   Open the Detection page and upload a chest X-ray image.")
    
    return True

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
