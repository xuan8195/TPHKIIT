"""
Test Grad-CAM functionality in Streamlit app
Run this to verify Grad-CAM is working before starting the full app.
"""

import sys
import numpy as np
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

def test_gradcam():
    """Test that Grad-CAM works with the loaded model"""
    print("=" * 60)
    print("  TESTING GRAD-CAM FUNCTIONALITY")
    print("=" * 60)
    print()
    
    try:
        # Import modules
        print("1. Importing modules...")
        from utils.model_loader import load_model
        from utils.visualization import generate_gradcam
        print("   ✅ Imports successful\n")
        
        # Load model
        print("2. Loading model...")
        model = load_model()
        if model is None:
            print("   ❌ Model failed to load")
            return False
        print("   ✅ Model loaded\n")
        
        # Create test image
        print("3. Creating test image...")
        test_image = np.random.rand(1, 150, 150, 1).astype('float32')
        print(f"   ✅ Test image created: {test_image.shape}\n")
        
        # Test prediction first
        print("4. Testing model prediction...")
        prediction = model.predict(test_image, verbose=0)
        print(f"   ✅ Prediction successful: {prediction[0][0]:.4f}\n")
        
        # Generate Grad-CAM
        print("5. Generating Grad-CAM...")
        heatmap, overlay = generate_gradcam(model, test_image, target_is_pneumonia=True)
        print(f"   ✅ Grad-CAM generated successfully!")
        print(f"   📊 Heatmap shape: {heatmap.shape}")
        print(f"   📊 Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        print(f"   📊 Overlay shape: {overlay.shape}")
        print(f"   📊 Overlay dtype: {overlay.dtype}\n")
        
        # Validate outputs
        print("6. Validating outputs...")
        assert heatmap.shape == (150, 150), f"❌ Heatmap shape incorrect: {heatmap.shape}"
        assert overlay.shape == (150, 150, 3), f"❌ Overlay shape incorrect: {overlay.shape}"
        assert overlay.dtype == np.uint8, f"❌ Overlay dtype incorrect: {overlay.dtype}"
        assert 0 <= heatmap.min() <= 1, f"❌ Heatmap min out of range: {heatmap.min()}"
        assert 0 <= heatmap.max() <= 1, f"❌ Heatmap max out of range: {heatmap.max()}"
        print("   ✅ All validations passed\n")
        
        print("=" * 60)
        print("  ✅ GRAD-CAM TEST PASSED!")
        print("=" * 60)
        print()
        print("🎉 Your Streamlit app is ready!")
        print()
        print("📝 To start the app:")
        print("   streamlit run Home.py")
        print()
        print("🔍 To test with a real X-ray:")
        print("   1. Start the app")
        print("   2. Go to Detection page")
        print("   3. Upload an X-ray image")
        print("   4. Click 'Analyze Image'")
        print("   5. See Grad-CAM overlay!")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n📋 Full traceback:")
        import traceback
        traceback.print_exc()
        print()
        print("💡 Common fixes:")
        print("   - Make sure model file exists: utils/pneumo_cnn.h5 or utils/pneumo_cnn.keras")
        print("   - Run from pneumonia-streamlit directory")
        print("   - Check all dependencies are installed")
        return False


if __name__ == "__main__":
    success = test_gradcam()
    sys.exit(0 if success else 1)
