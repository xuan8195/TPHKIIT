"""
Quick Test Script for Grad-CAM in Streamlit
This script helps verify that Grad-CAM is working correctly in your Streamlit app.
"""

import sys
from pathlib import Path
import numpy as np

# Add utils to path
sys.path.append(str(Path(__file__).parent / "pneumonia-streamlit"))

def test_gradcam_function():
    """Test that Grad-CAM function works correctly"""
    print("🔍 Testing Grad-CAM Functionality...\n")
    
    try:
        # Import required modules
        print("1️⃣ Importing modules...")
        from pneumonia-streamlit.utils.model_loader import load_model, preprocess_image
        from pneumonia-streamlit.utils.visualization import generate_gradcam
        print("   ✅ Imports successful\n")
        
        # Load model
        print("2️⃣ Loading model...")
        model = load_model()
        if model is None:
            print("   ❌ Model failed to load")
            return False
        print("   ✅ Model loaded successfully\n")
        
        # Create a dummy test image (150x150 grayscale)
        print("3️⃣ Creating test image...")
        test_image = np.random.rand(1, 150, 150, 1).astype('float32')
        print("   ✅ Test image created (shape:", test_image.shape, ")\n")
        
        # Generate Grad-CAM
        print("4️⃣ Generating Grad-CAM...")
        heatmap, overlay = generate_gradcam(model, test_image, target_is_pneumonia=True)
        print("   ✅ Grad-CAM generated successfully")
        print(f"   📊 Heatmap shape: {heatmap.shape}")
        print(f"   📊 Overlay shape: {overlay.shape}")
        print(f"   📊 Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        print(f"   📊 Overlay dtype: {overlay.dtype}\n")
        
        # Verify output shapes
        print("5️⃣ Verifying outputs...")
        assert heatmap.shape == (150, 150), f"Heatmap shape incorrect: {heatmap.shape}"
        assert overlay.shape == (150, 150, 3), f"Overlay shape incorrect: {overlay.shape}"
        assert overlay.dtype == np.uint8, f"Overlay dtype incorrect: {overlay.dtype}"
        assert 0 <= heatmap.min() <= 1, f"Heatmap values out of range"
        assert 0 <= heatmap.max() <= 1, f"Heatmap values out of range"
        print("   ✅ All validations passed\n")
        
        print("=" * 50)
        print("✅ GRAD-CAM TEST PASSED!")
        print("=" * 50)
        print("\n🎉 Your Streamlit app is ready to use Grad-CAM!\n")
        print("📝 Next steps:")
        print("   1. Run: streamlit run Home.py")
        print("   2. Navigate to Detection page")
        print("   3. Upload an X-ray image")
        print("   4. Click 'Analyze Image'")
        print("   5. See Grad-CAM overlay in results!\n")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("\n💡 Make sure you're running from the correct directory:")
        print("   cd 'c:\\Users\\Luo Yuxuan\\Desktop\\tp AAI\\TPHKIIT'")
        return False
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print(f"\n📋 Full error details:")
        import traceback
        traceback.print_exc()
        return False


def check_model_files():
    """Check if model files exist"""
    print("🔍 Checking model files...\n")
    
    model_dir = Path("pneumonia-streamlit/utils")
    h5_model = model_dir / "pneumo_cnn.h5"
    keras_model = model_dir / "pneumo_cnn.keras"
    
    print(f"Looking in: {model_dir.absolute()}\n")
    
    if h5_model.exists():
        size_mb = h5_model.stat().st_size / (1024 * 1024)
        print(f"✅ Found: pneumo_cnn.h5 ({size_mb:.2f} MB)")
    else:
        print(f"❌ Missing: pneumo_cnn.h5")
    
    if keras_model.exists():
        size_mb = keras_model.stat().st_size / (1024 * 1024)
        print(f"✅ Found: pneumo_cnn.keras ({size_mb:.2f} MB)")
    else:
        print(f"❌ Missing: pneumo_cnn.keras")
    
    print()
    
    if not (h5_model.exists() or keras_model.exists()):
        print("⚠️  No model files found!")
        print("\n💡 To copy model from notebook:")
        print("   1. Open: pneumonia-detection-using-cnn-92-6-accuracy.ipynb")
        print("   2. Run the last cell (converts and copies model)")
        print("   3. Model will be saved to pneumonia-streamlit/utils/\n")
        return False
    
    return True


def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...\n")
    
    required = {
        'streamlit': 'Streamlit',
        'tensorflow': 'TensorFlow',
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'PIL': 'Pillow'
    }
    
    all_installed = True
    
    for module, name in required.items():
        try:
            if module == 'cv2':
                import cv2
                version = cv2.__version__
            elif module == 'PIL':
                from PIL import Image
                version = Image.__version__ if hasattr(Image, '__version__') else 'installed'
            else:
                mod = __import__(module)
                version = mod.__version__
            
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: NOT INSTALLED")
            all_installed = False
    
    print()
    
    if not all_installed:
        print("⚠️  Some dependencies are missing!")
        print("\n💡 Install missing packages:")
        print("   pip install streamlit tensorflow numpy opencv-python pillow\n")
        return False
    
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("  GRAD-CAM STREAMLIT TEST")
    print("=" * 50)
    print()
    
    # Check dependencies
    deps_ok = check_dependencies()
    print("-" * 50)
    print()
    
    # Check model files
    model_ok = check_model_files()
    print("-" * 50)
    print()
    
    # Test Grad-CAM
    if deps_ok and model_ok:
        test_ok = test_gradcam_function()
        
        if test_ok:
            print("\n🎊 EVERYTHING LOOKS GOOD!")
            print("\n🚀 Ready to launch Streamlit app:")
            print("   cd pneumonia-streamlit")
            print("   streamlit run Home.py")
    else:
        print("\n⚠️  Please fix the issues above before testing Grad-CAM\n")
