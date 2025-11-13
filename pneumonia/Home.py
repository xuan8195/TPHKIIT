import streamlit as st
import sys
from pathlib import Path
import json
from streamlit_lottie import st_lottie

# Add utils to path
sys.path.append(str(Path(__file__).parent))

# Load Lottie animation
def load_lottie_file(filepath: str):
    """Load a Lottie animation from a JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading animation: {e}")
        return None

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #475569;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        display: inline-block;
        background: linear-gradient(135deg, #d4f1f4 0%, rgba(255, 255, 255, 0.95) 100%);
        padding: 2.5rem 2rem;
        margin: 1rem;
        border-radius: 20px;
        text-align: center;
        box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.08),
            0 2px 8px rgba(102, 126, 234, 0.15);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 320px;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .feature-card:hover {
        transform: translateY(-12px) scale(1.03);
        box-shadow:
            0 16px 48px rgba(0, 0, 0, 0.15),
            0 8px 20px rgba(102, 126, 234, 0.3);
        border: 1px solid #667eea;
    }
    
    .feature-card:hover .material-icons {
        transform: scale(1.15) rotate(5deg);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
    }
    
    .material-icons {
        font-size: 48px;
        color: #667eea;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #e8f0fe 0%, #f3e8ff 100%);
        width: 90px;
        height: 90px;
        margin: 0 auto 1.5rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1e3a8a;
    }
    
    .feature-desc {
        font-size: 1rem;
        line-height: 1.6;
        color: #64748b;
    }
    
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        text-align: center;
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        text-align: center;
    }
    
    .stats-label {
        font-size: 1rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        text-align: center;
    }
    
    .info-section {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 0.75rem;
        margin: 2rem 0;
    }
    
    .cta-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 0.5rem;
        text-decoration: none;
        font-weight: 600;
        display: inline-block;
        margin-top: 1rem;
    }
    
    .divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #667eea, transparent);
        margin: 3rem 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Enhanced Sidebar Stats */
    .sidebar-stat-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem 1rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 
            0 4px 12px rgba(0, 0, 0, 0.08),
            0 2px 6px rgba(102, 126, 234, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .sidebar-stat-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 
            0 12px 24px rgba(0, 0, 0, 0.12),
            0 6px 12px rgba(102, 126, 234, 0.2);
        border: 1px solid #667eea;
    }
    
    .sidebar-stat-icon {
        font-size: 32px;
        color: #667eea;
        background: linear-gradient(135deg, #e8f0fe 0%, #f3e8ff 100%);
        border-radius: 50%;
        width: 60px;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .sidebar-stat-card:hover .sidebar-stat-icon {
        transform: scale(1.15) rotate(5deg);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.3);
    }
    
    .sidebar-stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e3a8a;
        margin: 0.5rem 0 0.25rem 0;
    }
    
    .sidebar-stat-label {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<h1 class="main-header">Pneumonia Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Medical Image Analysis for Rapid Diagnosis</p>', unsafe_allow_html=True)

# Lottie Animation Section
lottie_covid = load_lottie_file("assets/Covid Icon _ Pneumonia.json")
if lottie_covid:
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st_lottie(
            lottie_covid,
            speed=1,
            reverse=False,
            loop=True,
            quality="high",
            height=500,
            width=None,
            key="covid_animation"
        )

st.markdown("<br>", unsafe_allow_html=True)

# Introduction Section
st.markdown("""
<div class="info-section">
<h2 style="color: #1e3a8a; margin-bottom: 1rem; text-align: center;">Welcome to Our Advanced Diagnostic Platform</h2>
<p style="font-size: 1.1rem; line-height: 1.8; color: #334155; text-align: center; max-width: 900px; margin: 0 auto;">
Our state-of-the-art Convolutional Neural Network (CNN) system provides rapid and accurate 
pneumonia detection from chest X-ray images. Leveraging deep learning technology, we assist 
healthcare professionals in making faster, more informed diagnostic decisions.
</p>
</div>
""", unsafe_allow_html=True)

# Stats Section
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

with col_stat1:
    st.markdown("""
    <div class="stats-card">
    <div class="stats-number">92.6%</div>
    <div class="stats-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col_stat2:
    st.markdown("""
    <div class="stats-card">
    <div class="stats-number">&lt;2s</div>
    <div class="stats-label">Analysis Time</div>
    </div>
    """, unsafe_allow_html=True)

with col_stat3:
    st.markdown("""
    <div class="stats-card">
    <div class="stats-number">99%</div>
    <div class="stats-label">Sensitivity</div>
    </div>
    """, unsafe_allow_html=True)

with col_stat4:
    st.markdown("""
    <div class="stats-card">
    <div class="stats-number">5,216</div>
    <div class="stats-label">Training Images</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Key Features Section
st.markdown("## Key Features")
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
    <span class="material-icons">biotech</span>
    <div class="feature-title">Instant Analysis</div>
    <div class="feature-desc">
    Upload chest X-ray images and receive instant pneumonia detection results 
    with detailed confidence scores.
    </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card" style="background: linear-gradient(135deg, #fce7f3 0%, rgba(255, 255, 255, 0.95) 100%);">
    <span class="material-icons" style="color: #ec4899; background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%);">insights</span>
    <div class="feature-title">Visual Analytics</div>
    <div class="feature-desc">
    Comprehensive visualizations including heatmaps, probability distributions, 
    and detailed performance metrics.
    </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card" style="background: linear-gradient(135deg, #dbeafe 0%, rgba(255, 255, 255, 0.95) 100%);">
    <span class="material-icons" style="color: #3b82f6; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);">verified</span>
    <div class="feature-title">High Sensitivity</div>
    <div class="feature-desc">
    Our CNN model achieves 92.6% accuracy with 99% sensitivity (pneumonia detection rate), 
    trained on 5,216 validated chest X-ray images.
    </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# How It Works Section
st.markdown("## How It Works")
st.markdown("<br>", unsafe_allow_html=True)

step_cols = st.columns(4)

with step_cols[0]:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
    <span class="material-icons" style="font-size: 48px; color: #667eea; background: linear-gradient(135deg, #e8f0fe 0%, #f3e8ff 100%); 
          border-radius: 50%; width: 70px; height: 70px; display: flex; align-items: center; justify-content: center; 
          margin: 0 auto 1rem; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);">upload_file</span>
    <h3 style="color: #1e3a8a; font-size: 1.2rem;">Upload Image</h3>
    <p style="color: #64748b;">Upload a chest X-ray image in JPG or PNG format</p>
    </div>
    """, unsafe_allow_html=True)

with step_cols[1]:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
    <span class="material-icons" style="font-size: 48px; color: #667eea; background: linear-gradient(135deg, #e8f0fe 0%, #f3e8ff 100%); 
          border-radius: 50%; width: 70px; height: 70px; display: flex; align-items: center; justify-content: center; 
          margin: 0 auto 1rem; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);">psychology</span>
    <h3 style="color: #1e3a8a; font-size: 1.2rem;">AI Processing</h3>
    <p style="color: #64748b;">Our CNN model analyzes the image for pneumonia indicators</p>
    </div>
    """, unsafe_allow_html=True)

with step_cols[2]:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
    <span class="material-icons" style="font-size: 48px; color: #667eea; background: linear-gradient(135deg, #e8f0fe 0%, #f3e8ff 100%); 
          border-radius: 50%; width: 70px; height: 70px; display: flex; align-items: center; justify-content: center; 
          margin: 0 auto 1rem; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);">assignment_turned_in</span>
    <h3 style="color: #1e3a8a; font-size: 1.2rem;">Get Results</h3>
    <p style="color: #64748b;">Receive instant diagnosis with confidence scores</p>
    </div>
    """, unsafe_allow_html=True)

with step_cols[3]:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
    <span class="material-icons" style="font-size: 48px; color: #667eea; background: linear-gradient(135deg, #e8f0fe 0%, #f3e8ff 100%); 
          border-radius: 50%; width: 70px; height: 70px; display: flex; align-items: center; justify-content: center; 
          margin: 0 auto 1rem; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);">analytics</span>
    <h3 style="color: #1e3a8a; font-size: 1.2rem;">View Analytics</h3>
    <p style="color: #64748b;">Explore detailed visualizations and insights</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# About Pneumonia Section
st.markdown("## About Pneumonia")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-section">
    <h3 style="color: #1e3a8a; margin-bottom: 1rem;">What is Pneumonia?</h3>
    <p style="line-height: 1.8; color: #334155;">
    Pneumonia is an infection that inflames the air sacs in one or both lungs. 
    The air sacs may fill with fluid or pus, causing cough with phlegm or pus, 
    fever, chills, and difficulty breathing. Early detection is crucial for 
    effective treatment and recovery.
    </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-section">
    <h3 style="color: #1e3a8a; margin-bottom: 1rem;">Why AI Detection?</h3>
    <p style="line-height: 1.8; color: #334155;">
    AI-powered detection systems can analyze chest X-rays in seconds, providing 
    consistent and reliable results. This technology assists radiologists in 
    making faster diagnoses, especially in resource-limited settings or during 
    high-volume periods.
    </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Call to Action
st.markdown("## Ready to Get Started?")
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style="text-align: center; background: white; padding: 2rem; border-radius: 1rem; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
    <h3 style="color: #1e3a8a; margin-bottom: 1rem;">Start Your Analysis Now</h3>
    <p style="color: #64748b; margin-bottom: 1.5rem;">
    Navigate to the Detection page to upload and analyze chest X-ray images, 
    or explore our Model Information to learn more about our technology.
    </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #94a3b8; border-top: 1px solid #e2e8f0; margin-top: 3rem;">
<p style="margin-bottom: 0.5rem;">Pneumonia Detection System | Powered by Deep Learning</p>
<p style="font-size: 0.875rem;">This tool is designed to assist medical professionals and should not replace professional medical advice.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Navigation")
    st.info("""
    Use the sidebar to navigate between different sections:
    
    **Detection**: Upload and analyze X-ray images
    
    **Analytics**: View detailed statistics and visualizations
    
    **Model Info**: Learn about our CNN architecture
    
    **About**: Information about the system
    """)
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Accuracy
    st.markdown("""
    <div class="sidebar-stat-card">
        <span class="material-icons sidebar-stat-icon">verified</span>
        <div class="sidebar-stat-value">92.6%</div>
        <div class="sidebar-stat-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sensitivity
    st.markdown("""
    <div class="sidebar-stat-card">
        <span class="material-icons sidebar-stat-icon">health_and_safety</span>
        <div class="sidebar-stat-value">99.0%</div>
        <div class="sidebar-stat-label">Sensitivity</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Processing Time
    st.markdown("""
    <div class="sidebar-stat-card">
        <span class="material-icons sidebar-stat-icon">speed</span>
        <div class="sidebar-stat-value">&lt; 2s</div>
        <div class="sidebar-stat-label">Processing Time</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Image Support
    st.markdown("""
    <div class="sidebar-stat-card">
        <span class="material-icons sidebar-stat-icon">image</span>
        <div class="sidebar-stat-value">JPG, PNG, JPEG</div>
        <div class="sidebar-stat-label">Image Support</div>
    </div>
    """, unsafe_allow_html=True)
