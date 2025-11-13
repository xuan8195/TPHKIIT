"""
Model Compatibility Checker

This script checks if your model is properly set up and compatible with the application.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def check_imports():
    """Check if all required packages are installed"""
    print("Checking required packages...")
    required_packages = {
        'streamlit': 'streamlit',
        'tensorflow': 'tensorflow',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'PIL': 'pillow',
        'plotly': 'plotly',
        'sklearn': 'scikit-learn'
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    else:
        print("\n✓ All required packages are installed!")
        return True

def check_model():
    """Check if model files exist"""
    print("\nChecking for model files...")
    
    model_path = Path(__file__).parent / "utils" / "model.pkl"
    weights_path = Path(__file__).parent / "utils" / "model_weights.h5"
    
    if model_path.exists():
        print(f"  ✓ Model file found: {model_path}")
        print(f"    Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
        return True
    elif weights_path.exists():
        print(f"  ✓ Model weights found: {weights_path}")
        print(f"    Size: {weights_path.stat().st_size / (1024*1024):.2f} MB")
        print("  ℹ The app will build the architecture and load these weights")
        return True
    else:
        print(f"  ✗ No model file found")
        print(f"\n  Expected locations:")
        print(f"    - {model_path}")
        print(f"    - {weights_path}")
        print(f"\n  ⚠️ The app will use the architecture without trained weights.")
        print(f"     Predictions may not be accurate until you add a trained model.")
        return False

def check_project_structure():
    """Check if all required files exist"""
    print("\nChecking project structure...")
    
    required_files = [
        'Home.py',
        'pages/1_Detection.py',
        'pages/2_Analytics.py',
        'pages/3_Model_Info.py',
        'pages/4_About.py',
        'utils/model_loader.py',
        'utils/visualization.py',
        'requirements.txt',
        '.streamlit/config.toml'
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            all_exist = False
    
    if all_exist:
        print("\n✓ All required files are present!")
    else:
        print("\n❌ Some files are missing!")
    
    return all_exist

def test_model_loading():
    """Try to load the model"""
    print("\nTesting model loading...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from utils.model_loader import load_model
        
        model = load_model()
        print("  ✓ Model loaded successfully!")
        
        # Try to get model summary
        try:
            import tensorflow as tf
            if isinstance(model, tf.keras.Model):
                print("  ✓ Model is a valid Keras model")
                print(f"  ℹ Model has {model.count_params():,} parameters")
        except:
            pass
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to load model: {str(e)}")
        print("  ℹ This is expected if you haven't added a trained model yet")
        return False

def main():
    print("=" * 60)
    print("Pneumonia Detection System - Compatibility Checker")
    print("=" * 60)
    print()
    
    # Run checks
    imports_ok = check_imports()
    structure_ok = check_project_structure()
    model_exists = check_model()
    
    if imports_ok and structure_ok:
        model_loads = test_model_loading()
    else:
        model_loads = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if imports_ok and structure_ok:
        print("✓ Project setup is complete!")
        print("✓ You can run the application with: streamlit run Home.py")
        
        if not model_exists or not model_loads:
            print("\n⚠️ WARNING: No trained model detected")
            print("   The app will run but predictions may not work correctly.")
            print("   To add a model:")
            print("   1. Train a model using the Kaggle notebook")
            print("   2. Save it as 'utils/model.pkl' or 'utils/model_weights.h5'")
        else:
            print("\n✓ Model is ready for predictions!")
    else:
        print("❌ Setup incomplete - please fix the issues above")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
