import streamlit as st
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_loader import get_model_summary

# Page configuration
st.set_page_config(
    page_title="Model Information",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    
    <style>
    .model-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    
    .architecture-card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .layer-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        margin: 0.5rem 0;
    }
    
    .spec-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .highlight-box {
        background: linear-gradient(135deg, #d4f1f4 0%, rgba(255, 255, 255, 0.95) 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 
            0 8px 24px rgba(0, 0, 0, 0.08),
            0 2px 8px rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .highlight-box:hover {
        transform: translateY(-8px);
        box-shadow: 
            0 12px 32px rgba(0, 0, 0, 0.12),
            0 4px 12px rgba(102, 126, 234, 0.2);
        border: 1px solid #667eea;
    }
    
    .highlight-box .material-icons {
        font-size: 48px;
        color: #667eea;
        background: linear-gradient(135deg, #e8f0fe 0%, #f3e8ff 100%);
        border-radius: 50%;
        width: 80px;
        height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .highlight-box:hover .material-icons {
        transform: scale(1.1) rotate(5deg);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.3);
    }
    
    .highlight-box h3 {
        color: #1e3a8a;
        margin-bottom: 1rem;
    }
    
    .highlight-box h2 {
        font-size: 2rem;
        margin: 0;
        color: #1e3a8a;
    }
    
    .highlight-box p {
        margin-top: 0.5rem;
        color: #64748b;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="model-header">Model Information</h1>', unsafe_allow_html=True)
st.markdown("Detailed information about the CNN architecture and training process")

# Model Overview
st.markdown("## Model Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="highlight-box">
    <span class="material-icons">psychology</span>
    <h3 style="margin-bottom: 1rem;">Architecture</h3>
    <h2 style="font-size: 2rem; margin: 0;">CNN</h2>
    <p style="margin-top: 0.5rem;">Convolutional Neural Network</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="highlight-box" style="background: linear-gradient(135deg, #fce7f3 0%, rgba(255, 255, 255, 0.95) 100%);">
    <span class="material-icons" style="color: #ec4899; background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%);">memory</span>
    <h3 style="margin-bottom: 1rem;">Parameters</h3>
    <h2 style="font-size: 2rem; margin: 0;">2.3M</h2>
    <p style="margin-top: 0.5rem;">Trainable Parameters</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="highlight-box" style="background: linear-gradient(135deg, #dbeafe 0%, rgba(255, 255, 255, 0.95) 100%);">
    <span class="material-icons" style="color: #3b82f6; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);">verified</span>
    <h3 style="margin-bottom: 1rem;">Accuracy</h3>
    <h2 style="font-size: 2rem; margin: 0;">92.6%</h2>
    <p style="margin-top: 0.5rem;">Test Set Performance</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Network Architecture
st.markdown("## Network Architecture")

model_summary = get_model_summary()

st.markdown("""
<div class="architecture-card">
<h3 style="color: #1e3a8a; margin-bottom: 1.5rem;">Layer-by-Layer Breakdown</h3>
""", unsafe_allow_html=True)

# Display architecture diagram
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("""
    ### Input Layer
    **Shape:** 150 x 150 x 1
    
    The model accepts grayscale chest X-ray images with dimensions of 150x150 pixels.
    """)

with col2:
    st.markdown("""
    ### Processing Pipeline
    
    The input image flows through 5 convolutional blocks with batch normalization and dropout,
    extracting hierarchical features from simple edges to complex pneumonia patterns.
    """)

st.markdown("### Convolutional Blocks")

# Block 1
st.markdown("""
<div class="layer-card">
<h4 style="margin-bottom: 0.5rem;">Block 1: Initial Feature Detection</h4>
<ul>
<li><strong>Conv2D Layer:</strong> 32 filters, 3x3 kernel, ReLU activation</li>
<li><strong>BatchNormalization:</strong> Normalizes activations</li>
<li><strong>MaxPooling2D:</strong> 2x2 pool size</li>
<li><strong>Output Shape:</strong> 75 x 75 x 32</li>
</ul>
<p style="margin-top: 1rem; opacity: 0.9;">Detects basic features like edges, lines, and textures</p>
</div>
""", unsafe_allow_html=True)

# Block 2
st.markdown("""
<div class="layer-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
<h4 style="margin-bottom: 0.5rem;">Block 2: Pattern Recognition</h4>
<ul>
<li><strong>Conv2D Layer:</strong> 64 filters, 3x3 kernel, ReLU activation</li>
<li><strong>Dropout:</strong> 10% dropout rate</li>
<li><strong>BatchNormalization:</strong> Normalizes activations</li>
<li><strong>MaxPooling2D:</strong> 2x2 pool size</li>
<li><strong>Output Shape:</strong> 37 x 37 x 64</li>
</ul>
<p style="margin-top: 1rem; opacity: 0.9;">Identifies more complex patterns and shapes in lung tissue</p>
</div>
""", unsafe_allow_html=True)

# Block 3
st.markdown("""
<div class="layer-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
<h4 style="margin-bottom: 0.5rem;">Block 3: Intermediate Features</h4>
<ul>
<li><strong>Conv2D Layer:</strong> 64 filters, 3x3 kernel, ReLU activation</li>
<li><strong>BatchNormalization:</strong> Normalizes activations</li>
<li><strong>MaxPooling2D:</strong> 2x2 pool size</li>
<li><strong>Output Shape:</strong> 18 x 18 x 64</li>
</ul>
<p style="margin-top: 1rem; opacity: 0.9;">Captures mid-level structural information</p>
</div>
""", unsafe_allow_html=True)

# Block 4
st.markdown("""
<div class="layer-card">
<h4 style="margin-bottom: 0.5rem;">Block 4: High-Level Features</h4>
<ul>
<li><strong>Conv2D Layer:</strong> 128 filters, 3x3 kernel, ReLU activation</li>
<li><strong>Dropout:</strong> 20% dropout rate</li>
<li><strong>BatchNormalization:</strong> Normalizes activations</li>
<li><strong>MaxPooling2D:</strong> 2x2 pool size</li>
<li><strong>Output Shape:</strong> 9 x 9 x 128</li>
</ul>
<p style="margin-top: 1rem; opacity: 0.9;">Extracts high-level diagnostic features</p>
</div>
""", unsafe_allow_html=True)

# Block 5
st.markdown("""
<div class="layer-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
<h4 style="margin-bottom: 0.5rem;">Block 5: Deep Feature Extraction</h4>
<ul>
<li><strong>Conv2D Layer:</strong> 256 filters, 3x3 kernel, ReLU activation</li>
<li><strong>Dropout:</strong> 20% dropout rate</li>
<li><strong>BatchNormalization:</strong> Normalizes activations</li>
<li><strong>MaxPooling2D:</strong> 2x2 pool size</li>
<li><strong>Output Shape:</strong> 4 x 4 x 256</li>
</ul>
<p style="margin-top: 1rem; opacity: 0.9;">Captures the most abstract pneumonia indicators</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Classification Layers")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="spec-card">
    <h4 style="color: #1e3a8a;">Flatten Layer</h4>
    <p>Converts 2D feature maps to 1D feature vector</p>
    <ul>
    <li><strong>Input:</strong> 4 x 4 x 256</li>
    <li><strong>Output:</strong> 4,096 features</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="spec-card">
    <h4 style="color: #1e3a8a;">Dropout Layer</h4>
    <p>Prevents overfitting during training</p>
    <ul>
    <li><strong>Rate:</strong> 0.2 (20%)</li>
    <li><strong>Purpose:</strong> Regularization</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="spec-card">
    <h4 style="color: #1e3a8a;">Dense Layer</h4>
    <p>Fully connected layer for feature integration</p>
    <ul>
    <li><strong>Units:</strong> 128</li>
    <li><strong>Activation:</strong> ReLU</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="spec-card">
    <h4 style="color: #1e3a8a;">Output Layer</h4>
    <p>Binary classification output</p>
    <ul>
    <li><strong>Units:</strong> 1</li>
    <li><strong>Activation:</strong> Sigmoid</li>
    <li><strong>Output:</strong> Probability (0-1)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Training Details
st.markdown("## Training Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="architecture-card">
    <h3 style="color: #1e3a8a; margin-bottom: 1rem;">Hyperparameters</h3>
    <table style="width: 100%; border-collapse: collapse;">
    <tr style="border-bottom: 1px solid #e2e8f0;">
        <td style="padding: 0.75rem; color: #64748b;">Optimizer</td>
        <td style="padding: 0.75rem; font-weight: 600; color: #1e3a8a;">RMSprop</td>
    </tr>
    <tr style="border-bottom: 1px solid #e2e8f0;">
        <td style="padding: 0.75rem; color: #64748b;">Learning Rate</td>
        <td style="padding: 0.75rem; font-weight: 600; color: #1e3a8a;">Adaptive (with ReduceLROnPlateau)</td>
    </tr>
    <tr style="border-bottom: 1px solid #e2e8f0;">
        <td style="padding: 0.75rem; color: #64748b;">Loss Function</td>
        <td style="padding: 0.75rem; font-weight: 600; color: #1e3a8a;">Binary Crossentropy</td>
    </tr>
    <tr style="border-bottom: 1px solid #e2e8f0;">
        <td style="padding: 0.75rem; color: #64748b;">Batch Size</td>
        <td style="padding: 0.75rem; font-weight: 600; color: #1e3a8a;">32</td>
    </tr>
    <tr style="border-bottom: 1px solid #e2e8f0;">
        <td style="padding: 0.75rem; color: #64748b;">Epochs</td>
        <td style="padding: 0.75rem; font-weight: 600; color: #1e3a8a;">12</td>
    </tr>
    <tr>
        <td style="padding: 0.75rem; color: #64748b;">Training Time</td>
        <td style="padding: 0.75rem; font-weight: 600; color: #1e3a8a;">~1.5 hours</td>
    </tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="architecture-card">
    <h3 style="color: #1e3a8a; margin-bottom: 1rem;">Data Augmentation</h3>
    <p style="color: #64748b; margin-bottom: 1rem;">
    Applied to increase dataset diversity and prevent overfitting:
    </p>
    <ul style="color: #334155;">
    <li><strong>Rotation:</strong> ±30 degrees</li>
    <li><strong>Width Shift:</strong> ±10%</li>
    <li><strong>Height Shift:</strong> ±10%</li>
    <li><strong>Zoom:</strong> ±20%</li>
    <li><strong>Horizontal Flip:</strong> Enabled</li>
    <li><strong>Vertical Flip:</strong> Disabled</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Technical Specifications
st.markdown("## Technical Specifications")

specs_col1, specs_col2, specs_col3 = st.columns(3)

with specs_col1:
    st.markdown("""
    <div class="spec-card">
    <h3 style="color: #1e3a8a;">Input Requirements</h3>
    <ul style="color: #334155;">
    <li>Image Format: JPG, PNG, JPEG</li>
    <li>Dimensions: 150x150 pixels</li>
    <li>Color Space: Grayscale (converted)</li>
    <li>Normalization: 0-1 range</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with specs_col2:
    st.markdown("""
    <div class="spec-card">
    <h3 style="color: #1e3a8a;">Output Format</h3>
    <ul style="color: #334155;">
    <li>Type: Binary Classification</li>
    <li>Classes: Normal (1), Pneumonia (0)</li>
    <li>Confidence Score: 0-1</li>
    <li>Threshold: 0.5</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with specs_col3:
    st.markdown("""
    <div class="spec-card">
    <h3 style="color: #1e3a8a;">Performance</h3>
    <ul style="color: #334155;">
    <li>Inference Time: < 2 seconds</li>
    <li>Accuracy: 92.6%</li>
    <li>Memory Usage: ~40 MB</li>
    <li>CPU/GPU Compatible: Yes</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Model Advantages
st.markdown("## Key Advantages")

adv_col1, adv_col2 = st.columns(2)

with adv_col1:
    st.success("""
    ### Strengths
    
    - **High Sensitivity**: 99% pneumonia detection rate
    - **Fast Inference**: Results in under 2 seconds
    - **Robust Architecture**: 5 convolutional blocks with regularization
    - **Clear Confidence**: Provides probability scores
    - **Efficient**: Optimized for production use
    - **Well-Trained**: 5,216 training images with augmentation
    """)

with adv_col2:
    st.info("""
    ### Use Cases
    
    - **Primary Screening**: Initial pneumonia detection
    - **Second Opinion**: Assist radiologists
    - **Resource-Limited Settings**: Where expert review is limited
    - **High-Volume Periods**: During epidemics or busy times
    - **Emergency Departments**: Quick triage tool
    - **Research**: Medical image analysis studies
    """)

st.markdown("---")

# Limitations and Disclaimer
st.markdown("## Important Considerations")

st.warning("""
### Limitations

While our model achieves high accuracy, please note:

- This tool is designed to **assist** medical professionals, not replace them
- Results should always be **verified** by qualified healthcare providers
- The model is trained on specific types of chest X-rays and may not generalize to all imaging conditions
- False positives and false negatives can occur
- Not suitable as the **sole diagnostic tool**
- Regular model updates and retraining are necessary to maintain performance
""")

st.error("""
### Medical Disclaimer

**This system is not a medical device and is not intended for diagnostic use without professional medical oversight.**

Always consult with qualified healthcare professionals for:
- Official diagnosis
- Treatment decisions
- Medical advice
- Clinical interpretation
""")

# Sidebar
with st.sidebar:
    st.markdown("### Model Details")
    
    st.info("""
    **Framework**: TensorFlow/Keras 2.10+
    
    **Model Version**: 1.0
    
    **Last Updated**: Nov 2025
    
    **Training Dataset**: Chest X-Ray Pneumonia (Kaggle)
    
    **Dataset Size**: 5,863 images
    """)
    
    st.markdown("---")
    st.markdown("### Quick Facts")
    st.metric("Total Layers", "19")
    st.metric("Conv Layers", "5")
    st.metric("Dense Layers", "2")
    st.metric("Parameters", "2.3M")
