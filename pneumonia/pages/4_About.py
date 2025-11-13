import streamlit as st

# Page configuration
st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .about-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    
    .section-card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .team-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 1rem;
        height: 100%;
    }
    
    .feature-list {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="about-header">About This System</h1>', unsafe_allow_html=True)
st.markdown("Learn more about the Pneumonia Detection System and its capabilities")

# Introduction
st.markdown("""
<div class="section-card">
<h2 style="color: #1e3a8a; margin-bottom: 1rem;">What is This System?</h2>
<p style="font-size: 1.1rem; line-height: 1.8; color: #334155;">
The Pneumonia Detection System is an advanced AI-powered medical imaging analysis tool designed to assist 
healthcare professionals in detecting pneumonia from chest X-ray images. Using a state-of-the-art Convolutional 
Neural Network (CNN) with 5 convolutional blocks, our system provides rapid, accurate, and consistent analysis 
to support clinical decision-making.
</p>
<p style="font-size: 1.1rem; line-height: 1.8; color: #334155; margin-top: 1rem;">
Built with cutting-edge deep learning technology and trained on 5,216 validated chest X-ray images from the 
Kaggle Chest X-Ray Pneumonia dataset, this system achieves 92.6% accuracy with an exceptional 99% sensitivity 
in detecting pneumonia cases, minimizing missed diagnoses.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Mission and Vision
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="section-card">
    <h2 style="color: #1e3a8a; margin-bottom: 1rem;">Our Mission</h2>
    <p style="line-height: 1.8; color: #334155;">
    To democratize access to advanced medical imaging analysis by providing healthcare 
    professionals with powerful AI tools that enhance diagnostic accuracy, speed, and 
    accessibility, ultimately improving patient outcomes worldwide.
    </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="section-card">
    <h2 style="color: #1e3a8a; margin-bottom: 1rem;">Our Vision</h2>
    <p style="line-height: 1.8; color: #334155;">
    A world where every healthcare facility, regardless of location or resources, has 
    access to AI-powered diagnostic tools that support medical professionals in delivering 
    timely, accurate diagnoses and better patient care.
    </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Key Features
st.markdown("## Key Features & Capabilities")

feature_col1, feature_col2, feature_col3 = st.columns(3)

with feature_col1:
    st.markdown("""
    <div class="feature-list">
    <h3 style="color: #1e3a8a;">AI Technology</h3>
    <ul style="color: #334155;">
    <li>5-Layer CNN Architecture</li>
    <li>92.6% Overall Accuracy</li>
    <li>99% Sensitivity (Pneumonia Detection)</li>
    <li>Real-time Image Processing (< 2s)</li>
    <li>Batch Normalization & Dropout</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with feature_col2:
    st.markdown("""
    <div class="feature-list">
    <h3 style="color: #1e3a8a;">User Experience</h3>
    <ul style="color: #334155;">
    <li>Intuitive Streamlit Interface</li>
    <li>Instant Results (< 2 seconds)</li>
    <li>Interactive Plotly Visualizations</li>
    <li>Comprehensive Performance Analytics</li>
    <li>Downloadable Reports</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with feature_col3:
    st.markdown("""
    <div class="feature-list">
    <h3 style="color: #1e3a8a;">Clinical Support</h3>
    <ul style="color: #334155;">
    <li>Probability-Based Confidence Scoring</li>
    <li>Detailed Performance Metrics</li>
    <li>Evidence-Based Results (AUC: 0.973)</li>
    <li>Confusion Matrix Analysis</li>
    <li>Clinical Decision Support</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Technology Stack
st.markdown("## Technology Stack")

st.markdown("""
<div class="section-card">
<h3 style="color: #1e3a8a; margin-bottom: 1.5rem;">Built With Modern Technologies</h3>
""", unsafe_allow_html=True)

tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)

with tech_col1:
    st.markdown("""
    **Machine Learning**
    - TensorFlow
    - Keras
    - NumPy
    - scikit-learn
    """)

with tech_col2:
    st.markdown("""
    **Web Framework**
    - Streamlit 1.28+
    - Python 3.9-3.12
    - Plotly 5.0+
    - PIL/Pillow
    """)

with tech_col3:
    st.markdown("""
    **Data Processing**
    - Pandas
    - OpenCV
    - Image Processing
    - Data Augmentation
    """)

with tech_col4:
    st.markdown("""
    **Visualization**
    - Plotly Charts
    - Interactive Graphs
    - Custom CSS
    - Responsive Design
    """)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# How It Helps
st.markdown("## How This System Helps Healthcare Professionals")

help_col1, help_col2 = st.columns(2)

with help_col1:
    st.success("""
    ### For Radiologists & Physicians
    
    - **High Sensitivity**: 99% pneumonia detection rate minimizes missed cases
    - **Quick Screening**: Initial triage tool for urgent cases
    - **Consistency**: Maintains objective evaluation criteria
    - **Documentation**: Generates detailed analysis with confidence scores
    - **Second Opinion**: Provides AI-assisted confirmation
    - **Workload Management**: Prioritizes high-risk cases
    """)

with help_col2:
    st.info("""
    ### For Healthcare Facilities
    
    - **Resource Optimization**: Effective in resource-limited settings
    - **24/7 Availability**: Continuous analysis capability
    - **Quality Assurance**: Helps maintain diagnostic standards
    - **Training Tool**: Educational resource for medical students
    - **Cost-Effective**: Reduces diagnostic delays
    - **Scalable**: Handles high volume efficiently
    """)

st.markdown("---")

# Clinical Workflow Integration
st.markdown("## Clinical Workflow Integration")

st.markdown("""
<div class="section-card">
<h3 style="color: #1e3a8a; margin-bottom: 1rem;">Seamless Integration into Existing Workflows</h3>
<p style="line-height: 1.8; color: #334155; margin-bottom: 1.5rem;">
Our system is designed to complement, not replace, existing clinical workflows. Here's how it fits into 
the diagnostic process:
</p>
""", unsafe_allow_html=True)

workflow_cols = st.columns(5)

with workflow_cols[0]:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
    <div style="background: #667eea; color: white; width: 50px; height: 50px; 
                border-radius: 50%; display: flex; align-items: center; 
                justify-content: center; margin: 0 auto 0.5rem; font-weight: 700;">1</div>
    <p style="color: #64748b; font-size: 0.9rem;">X-ray Acquisition</p>
    </div>
    """, unsafe_allow_html=True)

with workflow_cols[1]:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
    <div style="background: #667eea; color: white; width: 50px; height: 50px; 
                border-radius: 50%; display: flex; align-items: center; 
                justify-content: center; margin: 0 auto 0.5rem; font-weight: 700;">2</div>
    <p style="color: #64748b; font-size: 0.9rem;">AI Analysis</p>
    </div>
    """, unsafe_allow_html=True)

with workflow_cols[2]:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
    <div style="background: #667eea; color: white; width: 50px; height: 50px; 
                border-radius: 50%; display: flex; align-items: center; 
                justify-content: center; margin: 0 auto 0.5rem; font-weight: 700;">3</div>
    <p style="color: #64748b; font-size: 0.9rem;">Expert Review</p>
    </div>
    """, unsafe_allow_html=True)

with workflow_cols[3]:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
    <div style="background: #667eea; color: white; width: 50px; height: 50px; 
                border-radius: 50%; display: flex; align-items: center; 
                justify-content: center; margin: 0 auto 0.5rem; font-weight: 700;">4</div>
    <p style="color: #64748b; font-size: 0.9rem;">Final Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

with workflow_cols[4]:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
    <div style="background: #667eea; color: white; width: 50px; height: 50px; 
                border-radius: 50%; display: flex; align-items: center; 
                justify-content: center; margin: 0 auto 0.5rem; font-weight: 700;">5</div>
    <p style="color: #64748b; font-size: 0.9rem;">Treatment Plan</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Research and Development
st.markdown("## Research & Development")

rd_col1, rd_col2 = st.columns(2)

with rd_col1:
    st.markdown("""
    <div class="section-card">
    <h3 style="color: #1e3a8a;">Data Sources</h3>
    <p style="color: #334155; line-height: 1.8;">
    Our model is trained on the Chest X-Ray Pneumonia dataset from Kaggle:
    </p>
    <ul style="color: #334155;">
    <li><strong>Kaggle Chest X-Ray Pneumonia Dataset</strong></li>
    <li>Source: Guangzhou Women and Children's Medical Center</li>
    <li>Total Images: 5,863 X-rays</li>
    <li>Training: 5,216 images</li>
    <li>Testing: 624 images</li>
    <li>Validated by expert physicians</li>
    <li>Pediatric patients (1-5 years old)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with rd_col2:
    st.markdown("""
    <div class="section-card">
    <h3 style="color: #1e3a8a;">Training Process</h3>
    <p style="color: #334155; line-height: 1.8;">
    Rigorous training and validation process:
    </p>
    <ul style="color: #334155;">
    <li>12 training epochs</li>
    <li>Data augmentation (rotation, zoom, shift, flip)</li>
    <li>RMSprop optimizer with learning rate reduction</li>
    <li>Binary crossentropy loss function</li>
    <li>Batch normalization for stability</li>
    <li>Dropout regularization (10-20%)</li>
    <li>Independent test set evaluation</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Disclaimer
st.markdown("## Important Information")

st.error("""
### Medical Disclaimer

**IMPORTANT: This system is not a substitute for professional medical advice, diagnosis, or treatment.**

- This tool is designed to **assist** healthcare professionals, not replace them
- All results must be **reviewed and confirmed** by qualified medical professionals
- Do not use this system as the **sole basis** for diagnosis or treatment decisions
- Always seek the advice of qualified healthcare providers with any questions regarding medical conditions
- Never disregard professional medical advice or delay seeking it based on this tool's results
- In case of emergency, contact your local emergency services immediately

This system is intended for **informational and educational purposes** and as a **decision support tool** 
for qualified healthcare professionals only.
""")

st.markdown("---")

# Contact and Support
st.markdown("## Contact & Support")

contact_col1, contact_col2, contact_col3 = st.columns(3)

with contact_col1:
    st.markdown("""
    <div class="section-card">
    <h3 style="color: #1e3a8a;">Project Information</h3>
    <p style="color: #334155;">
    This is an academic research project for pneumonia detection using deep learning.
    </p>
    <p style="color: #667eea; font-weight: 600;">
    GitHub: github.com/pneumonia-detection
    </p>
    </div>
    """, unsafe_allow_html=True)

with contact_col2:
    st.markdown("""
    <div class="section-card">
    <h3 style="color: #1e3a8a;">Technical Details</h3>
    <p style="color: #334155;">
    For model architecture and implementation details:
    </p>
    <p style="color: #667eea; font-weight: 600;">
    See Model Info and Analytics pages
    </p>
    </div>
    """, unsafe_allow_html=True)

with contact_col3:
    st.markdown("""
    <div class="section-card">
    <h3 style="color: #1e3a8a;">Dataset Source</h3>
    <p style="color: #334155;">
    Original dataset available at:
    </p>
    <p style="color: #667eea; font-weight: 600;">
    Kaggle: Chest X-Ray Pneumonia
    </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Version Information
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #94a3b8; border-top: 1px solid #e2e8f0;">
<p style="margin-bottom: 0.5rem;"><strong>Pneumonia Detection System v1.0</strong></p>
<p style="font-size: 0.875rem;">Powered by Deep Learning CNN | November 2025</p>
<p style="font-size: 0.875rem; margin-top: 1rem;">Academic Research Project | Educational Purposes</p>
<p style="font-size: 0.875rem; margin-top: 0.5rem;">Dataset: Kaggle Chest X-Ray Pneumonia (Guangzhou Medical Center)</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Quick Links")
    
    st.markdown("""
    - [Documentation](#)
    - [API Reference](#)
    - [Privacy Policy](#)
    - [Terms of Service](#)
    """)
    
    st.markdown("---")
    st.markdown("### System Version")
    st.info("""
    **Version**: 1.0
    
    **Release Date**: November 2025
    
    **Model Accuracy**: 92.6%
    
    **Sensitivity**: 99.0%
    
    **Framework**: TensorFlow/Keras 2.10+
    """)
    
    st.markdown("---")
    st.markdown("### Acknowledgments")
    st.markdown("""
    <div style="font-size: 0.875rem; color: #64748b;">
    <p>Dataset: Kermany et al., Guangzhou Women and Children's Medical Center</p>
    <p>Built with TensorFlow, Keras, Streamlit, and Plotly</p>
    <p>Thanks to the open-source community and medical research contributors</p>
    </div>
    """, unsafe_allow_html=True)
