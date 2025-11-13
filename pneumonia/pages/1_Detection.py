import streamlit as st
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import io
import time

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_loader import load_model, preprocess_image, predict
from utils.visualization import (
    plot_prediction_bars,
    create_confidence_gauge,
    plot_image_with_overlay,
    generate_gradcam,
)

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    
    <style>
    .detection-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .prediction-positive {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    .prediction-negative {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #92400e;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #dbeafe;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #1e40af;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #065f46;
        margin: 1rem 0;
    }
    
    /* Enhanced metric container for educational cards */
    .metric-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 
            0 4px 12px rgba(0, 0, 0, 0.08),
            0 1px 4px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
        border: 2px solid transparent;
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 280px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
    }
    
    .metric-container:hover {
        transform: translateY(-12px) scale(1.02);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.15),
            0 8px 16px rgba(102, 126, 234, 0.2);
        border: 2px solid #667eea;
    }
    
    .metric-icon {
        font-size: 48px;
        color: #667eea;
        background: linear-gradient(135deg, #e8f0fe 0%, #f3e8ff 100%);
        border-radius: 50%;
        width: 80px;
        height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1.5rem;
        box-shadow: 
            0 8px 16px rgba(102, 126, 234, 0.2),
            0 0 0 8px rgba(102, 126, 234, 0.05);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-container:hover .metric-icon {
        transform: scale(1.2) rotate(10deg);
        box-shadow: 
            0 12px 24px rgba(102, 126, 234, 0.3),
            0 0 0 12px rgba(102, 126, 234, 0.08);
    }
    
    .metric-label {
        font-size: 1.1rem;
        color: #1e3a8a;
        font-weight: 600;
        text-transform: none;
        letter-spacing: normal;
        margin-bottom: 0.75rem;
    }
    
    .section-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin: 3rem 0 2rem 0;
        padding-bottom: 1rem;
        border-bottom: 3px solid #e5e7eb;
    }
    
    .section-header h3 {
        color: #1e3a8a;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    
    .section-icon {
        font-size: 2rem;
        color: #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def get_model():
    try:
        return load_model()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

try:
    model = get_model()
    model_loaded = model is not None
    
    # Check if model has been trained
    if model_loaded:
        # Try a dummy prediction to see if weights are loaded
        try:
            import numpy as np
            dummy_input = np.random.rand(1, 150, 150, 1)  # Fixed: model expects 1 channel (grayscale)
            _ = model.predict(dummy_input, verbose=0)
            model_trained = True
        except:
            model_trained = False
    else:
        model_trained = False
except Exception as e:
    model_loaded = False
    model_trained = False
    st.error(f"Failed to load model: {str(e)}")

# Header
st.markdown('<h1 class="detection-header">Pneumonia Detection</h1>', unsafe_allow_html=True)
st.markdown("Upload a chest X-ray image for instant AI-powered analysis")

# Show warning if model is not trained
if model_loaded and not model_trained:
    st.warning("""
    **⚠️ Model Not Trained**: The model architecture is loaded but doesn't have trained weights yet.
    
    To use the trained model from the Kaggle notebook:
    1. Download the trained model weights (`.h5` file) or saved model
    2. Place it in the `utils/` folder as either:
       - `model_weights.h5` (just the weights)
       - `model.h5` (complete saved model)
    3. Restart the application
    
    For now, predictions will be random and not accurate.
    """)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

# Upload Section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### Upload Chest X-Ray Image")
st.markdown("Supported formats: JPG, JPEG, PNG")
uploaded_file = st.file_uploader("Upload X-Ray Image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed", key=f"file_uploader_{st.session_state.file_uploader_key}")
st.markdown('</div>', unsafe_allow_html=True)

# Main content
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Original Image")
        image = Image.open(uploaded_file)
        st.image(image, width='stretch')
        
        # Image information
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"""
        **Image Details:**
        - Size: {image.size[0]} x {image.size[1]} pixels
        - Format: {image.format}
        - Mode: {image.mode}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Analysis")
        
        if model_loaded:
            # Analyze button
            if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image... Please wait"):
                    # Progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Preprocess and predict
                    processed_image = preprocess_image(image)
                    prediction, confidence = predict(model, processed_image)
                    
                    # Store results in session state
                    st.session_state.prediction_made = True
                    st.session_state.prediction_result = {
                        'prediction': prediction,
                        'confidence': confidence,
                        'image': processed_image
                    }
                    
                    st.rerun()
            
            # Display prediction results in col2
            if st.session_state.prediction_made and st.session_state.prediction_result:
                result = st.session_state.prediction_result
                prediction = result['prediction']
                confidence = result['confidence']
                
                # Prediction result
                if prediction == 'PNEUMONIA':
                    st.markdown(f'<div class="prediction-positive">⚠️ PNEUMONIA DETECTED</div>', unsafe_allow_html=True)
                    st.markdown(f"**Confidence Score:** {confidence:.2%}")
                    
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown("""
                    **Important Notice:**
                    - This result suggests the presence of pneumonia
                    - Please consult with a qualified healthcare professional
                    - Further diagnostic tests may be required
                    - Do not use this as the sole basis for treatment decisions
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-negative">✓ NO PNEUMONIA DETECTED</div>', unsafe_allow_html=True)
                    st.markdown(f"**Confidence Score:** {confidence:.2%}")
                    
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown("""
                    **Result Interpretation:**
                    - No signs of pneumonia detected in this X-ray
                    - This is an AI-assisted screening tool
                    - If symptoms persist, consult a healthcare professional
                    - Regular check-ups are recommended
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Model not loaded. Please check your model file.")
    
    # Educational cards and details OUTSIDE columns (full width)
    if st.session_state.prediction_made and st.session_state.prediction_result:
        st.markdown("---")
        
        result = st.session_state.prediction_result
        prediction = result['prediction']
        confidence = result['confidence']
        
        # Show educational cards only for PNEUMONIA
        if prediction == 'PNEUMONIA':
            # Next steps & educational cards
            st.markdown("""
            <div class="section-header">
                <span class="material-icons section-icon">assignment</span>
                <h3>Next Steps & Recommendations</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col_n1, col_n2, col_n3 = st.columns(3)
            
            with col_n1:
                st.markdown("""
                <div class="metric-container" style="background: linear-gradient(135deg, #e0f2fe 0%, #ffffff 100%);">
                    <span class="material-icons metric-icon" style="color: #0284c7; background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%); box-shadow: 0 8px 16px rgba(2, 132, 199, 0.2), 0 0 0 8px rgba(2, 132, 199, 0.05);">medical_services</span>
                    <div class="metric-label">See a Doctor</div>
                    <p style="color: #64748b; font-size: 0.95rem; line-height: 1.6; margin-top: 0.5rem;">
                    Seek urgent medical advice — bring this X-ray and describe any symptoms to your clinician for proper diagnosis.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_n2:
                st.markdown("""
                <div class="metric-container" style="background: linear-gradient(135deg, #fce7f3 0%, #ffffff 100%);">
                    <span class="material-icons metric-icon" style="color: #ec4899; background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%); box-shadow: 0 8px 16px rgba(236, 72, 153, 0.2), 0 0 0 8px rgba(236, 72, 153, 0.05);">sick</span>
                    <div class="metric-label">Common Symptoms</div>
                    <p style="color: #64748b; font-size: 0.95rem; line-height: 1.6; margin-top: 0.5rem;">
                    Watch for: persistent cough, high fever, difficulty breathing, and chest pain. Seek immediate care if symptoms worsen.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_n3:
                st.markdown("""
                <div class="metric-container" style="background: linear-gradient(135deg, #dcfce7 0%, #ffffff 100%);">
                    <span class="material-icons metric-icon" style="color: #22c55e; background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); box-shadow: 0 8px 16px rgba(34, 197, 94, 0.2), 0 0 0 8px rgba(34, 197, 94, 0.05);">health_and_safety</span>
                    <div class="metric-label">Prevention & Resources</div>
                    <p style="color: #64748b; font-size: 0.95rem; line-height: 1.6; margin-top: 0.5rem;">
                    Get vaccinated, maintain good hygiene, avoid smoking. Learn more: <a href="https://www.who.int/health-topics/pneumonia" target="_blank" style="color: #22c55e; font-weight: 600;">WHO</a>, <a href="https://www.moh.gov.sg" target="_blank" style="color: #22c55e; font-weight: 600;">MOH</a>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="section-header">
                <span class="material-icons section-icon">shield</span>
                <h3>How to Reduce Your Risk</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            
            with col_r1:
                st.markdown("""
                <div class="metric-container" style="background: linear-gradient(135deg, #fef3c7 0%, #ffffff 100%); min-height: 260px;">
                    <span class="material-icons metric-icon" style="color: #f59e0b; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); box-shadow: 0 8px 16px rgba(245, 158, 11, 0.2), 0 0 0 8px rgba(245, 158, 11, 0.05);">restaurant</span>
                    <div class="metric-label">Healthy Diet</div>
                    <p style="color: #64748b; font-size: 0.9rem; line-height: 1.6; margin-top: 0.5rem;">
                    Eat plenty of fruits, vegetables, and whole grains to strengthen your immune system.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_r2:
                st.markdown("""
                <div class="metric-container" style="background: linear-gradient(135deg, #dbeafe 0%, #ffffff 100%); min-height: 260px;">
                    <span class="material-icons metric-icon" style="color: #3b82f6; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); box-shadow: 0 8px 16px rgba(59, 130, 246, 0.2), 0 0 0 8px rgba(59, 130, 246, 0.05);">directions_run</span>
                    <div class="metric-label">Exercise</div>
                    <p style="color: #64748b; font-size: 0.9rem; line-height: 1.6; margin-top: 0.5rem;">
                    Get at least 30 minutes of moderate physical activity daily for respiratory health.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_r3:
                st.markdown("""
                <div class="metric-container" style="background: linear-gradient(135deg, #fee2e2 0%, #ffffff 100%); min-height: 260px;">
                    <span class="material-icons metric-icon" style="color: #ef4444; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); box-shadow: 0 8px 16px rgba(239, 68, 68, 0.2), 0 0 0 8px rgba(239, 68, 68, 0.05);">smoke_free</span>
                    <div class="metric-label">Quit Smoking</div>
                    <p style="color: #64748b; font-size: 0.9rem; line-height: 1.6; margin-top: 0.5rem;">
                    Smoking damages lungs. Seek support groups or medical advice to quit successfully.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_r4:
                st.markdown("""
                <div class="metric-container" style="background: linear-gradient(135deg, #e0e7ff 0%, #ffffff 100%); min-height: 260px;">
                    <span class="material-icons metric-icon" style="color: #6366f1; background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); box-shadow: 0 8px 16px rgba(99, 102, 241, 0.2), 0 0 0 8px rgba(99, 102, 241, 0.05);">self_improvement</span>
                    <div class="metric-label">Stress Management</div>
                    <p style="color: #64748b; font-size: 0.9rem; line-height: 1.6; margin-top: 0.5rem;">
                    Practice mindfulness, meditation, or yoga to reduce stress and boost immunity.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Confidence metrics
        st.markdown("### Confidence Breakdown")
        pneumonia_conf = confidence if prediction == 'PNEUMONIA' else 1 - confidence
        normal_conf = 1 - pneumonia_conf
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Pneumonia", f"{pneumonia_conf:.2%}")
        with col_m2:
            st.metric("Normal", f"{normal_conf:.2%}")
    
    # Detailed Analysis Section
    if st.session_state.prediction_made and st.session_state.prediction_result:
        st.markdown("---")
        st.markdown("## Detailed Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Probability Distribution", "Confidence Gauge", "Image Comparison"])
        
        with tab1:
            st.markdown("### Prediction Probability")
            result = st.session_state.prediction_result
            prediction = result['prediction']
            confidence = result['confidence']
            
            pneumonia_prob = confidence if prediction == 'PNEUMONIA' else 1 - confidence
            normal_prob = 1 - pneumonia_prob
            
            fig = plot_prediction_bars(pneumonia_prob, normal_prob)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            This chart shows the probability distribution of the model's prediction. 
            Higher values indicate stronger confidence in that diagnosis.
            """)
        
        with tab2:
            st.markdown("### Confidence Gauge")
            fig = create_confidence_gauge(confidence)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            The gauge displays the model's confidence level in its prediction:
            - **0-50%**: Low confidence - Further examination recommended
            - **50-80%**: Moderate confidence - Consider additional tests
            - **80-100%**: High confidence - Strong diagnostic indicator
            """)
        
        with tab3:
            st.markdown("### Processed Image Comparison")
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**Original Image**")
                st.image(image, width='stretch')
            
            with col_b:
                st.markdown("**Processed for Analysis**")
                # Display preprocessed image
                processed_display = result['image'][0] if len(result['image'].shape) == 4 else result['image']
                if processed_display.shape[-1] == 1:
                    processed_display = processed_display.squeeze()
                st.image(processed_display, width='stretch', clamp=True)

                # Grad-CAM overlay (if model available)
                try:
                    st.markdown("**Grad-CAM Overlay**")
                    # Determine target for Grad-CAM: True => pneumonia signal
                    target_pneumonia = True if prediction == 'PNEUMONIA' else False
                    heatmap, overlay = generate_gradcam(model, result['image'], target_is_pneumonia=target_pneumonia)

                    # OpenCV returns BGR; convert to RGB for display
                    import cv2
                    import numpy as np
                    from PIL import Image
                    from io import BytesIO

                    if overlay is not None:
                        # If overlay dtype isn't uint8, cast it
                        if overlay.dtype != np.uint8:
                            overlay_disp = np.clip(overlay, 0, 255).astype('uint8')
                        else:
                            overlay_disp = overlay

                        # Convert BGR (OpenCV) to RGB
                        try:
                            overlay_rgb = cv2.cvtColor(overlay_disp, cv2.COLOR_BGR2RGB)
                        except Exception:
                            overlay_rgb = overlay_disp

                        st.image(overlay_rgb, width='stretch', caption="Red/yellow regions indicate areas that most influenced the model's decision.")

                        # Offer download of the overlay image
                        buf = BytesIO()
                        Image.fromarray(overlay_rgb).save(buf, format='PNG')
                        buf.seek(0)
                        st.download_button(
                            label="📥 Download Overlay PNG",
                            data=buf,
                            file_name="gradcam_overlay.png",
                            mime="image/png"
                        )
                except Exception as e:
                    st.warning(f"Grad-CAM not available: {str(e)}")
            
            st.info("""
            The image is preprocessed to match the model's training data:
            - Resized to 150x150 pixels
            - Normalized pixel values
            - Converted to grayscale if needed
            """)
    
    # Action buttons
    if st.session_state.prediction_made:
        st.markdown("---")
        col_b1, col_b2, col_b3 = st.columns(3)
        
        with col_b1:
            if st.button("🔄 Analyze Another Image", use_container_width=True):
                st.session_state.prediction_made = False
                st.session_state.prediction_result = None
                # Clear the uploaded file by using a key that changes
                if 'file_uploader_key' not in st.session_state:
                    st.session_state.file_uploader_key = 0
                st.session_state.file_uploader_key += 1
                st.rerun()
        
        with col_b2:
            # Export results button
            if st.button("📥 Export Results", use_container_width=True):
                result = st.session_state.prediction_result
                report = f"""
PNEUMONIA DETECTION REPORT
========================

Prediction: {result['prediction']}
Confidence: {result['confidence']:.2%}

Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

DISCLAIMER:
This is an AI-assisted diagnostic tool and should not be used as 
the sole basis for medical decisions. Please consult with a 
qualified healthcare professional for proper diagnosis and treatment.
                """
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"pneumonia_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        with col_b3:
            if st.button("📊 View Analytics", use_container_width=True):
                st.switch_page("pages/2_Analytics.py")

else:
    # Instructions when no image is uploaded
    st.markdown("---")
    st.markdown("## How to Use This Tool")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3 style="color: #1e3a8a;">1. Prepare Image</h3>
        <p style="color: #64748b;">
        Ensure you have a clear chest X-ray image in JPG, JPEG, or PNG format. 
        The image should be properly exposed and positioned.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3 style="color: #1e3a8a;">2. Upload & Analyze</h3>
        <p style="color: #64748b;">
        Click the upload button above to select your X-ray image. 
        Then click 'Analyze Image' to start the detection process.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3 style="color: #1e3a8a;">3. Review Results</h3>
        <p style="color: #64748b;">
        Examine the detection results, confidence scores, and detailed 
        visualizations to understand the analysis.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("""
    **Important Guidelines:**
    - Use high-quality chest X-ray images for best results
    - Frontal (PA or AP) views work best
    - Ensure the entire chest cavity is visible
    - Avoid images with excessive noise or artifacts
    - This tool is for screening purposes only and should not replace professional medical diagnosis
    """)

# Sidebar
with st.sidebar:
    st.markdown("### Detection Settings")
    
    st.markdown("**Model Information**")
    st.info("""
    - Architecture: CNN
    - Accuracy: 96%
    - Training Images: 5,000+
    - Classes: Normal, Pneumonia
    """)
    
    st.markdown("---")
    st.markdown("### Need Help?")
    st.markdown("""
    - Ensure images are clear and properly formatted
    - Check that the X-ray shows the full chest
    - For technical issues, contact support
    """)
