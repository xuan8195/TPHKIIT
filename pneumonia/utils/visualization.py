import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def plot_prediction_bars(pneumonia_prob, normal_prob):
    """
    Create a bar chart showing prediction probabilities
    
    Args:
        pneumonia_prob: Probability of pneumonia
        normal_prob: Probability of normal
        
    Returns:
        plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Normal', 'Pneumonia'],
        y=[normal_prob, pneumonia_prob],
        marker=dict(
            color=['#4facfe', '#f5576c'],
            line=dict(color='white', width=2)
        ),
        text=[f'{normal_prob:.2%}', f'{pneumonia_prob:.2%}'],
        textposition='auto',
        textfont=dict(size=16, color='white', family='Arial Black')
    ))
    
    fig.update_layout(
        title={
            'text': 'Prediction Probability Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1e3a8a', 'family': 'Arial'}
        },
        xaxis_title='Classification',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    
    return fig

def create_confidence_gauge(confidence):
    """
    Create a gauge chart showing confidence level
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level (%)", 'font': {'size': 24, 'color': '#1e3a8a'}},
        delta={'reference': 80, 'increasing': {'color': "#10b981"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#1e3a8a"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 50], 'color': '#fee2e2'},
                {'range': [50, 80], 'color': '#fef3c7'},
                {'range': [80, 100], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1e3a8a", 'family': "Arial"},
        height=400
    )
    
    return fig

def plot_image_with_overlay(image_array):
    """
    Display image with optional heatmap overlay
    
    Args:
        image_array: Image as numpy array
        
    Returns:
        plotly figure
    """
    if len(image_array.shape) == 4:
        image_array = image_array[0]
    
    fig = px.imshow(image_array, color_continuous_scale='gray')
    
    fig.update_layout(
        title={
            'text': 'Processed X-Ray Image',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1e3a8a'}
        },
        coloraxis_showscale=False,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return fig

def create_metrics_chart(metrics_data):
    """
    Create a chart displaying multiple metrics
    
    Args:
        metrics_data: Dictionary with metric names and values
        
    Returns:
        plotly figure
    """
    fig = go.Figure()
    
    metrics = list(metrics_data.keys())
    values = list(metrics_data.values())
    
    fig.add_trace(go.Bar(
        x=metrics,
        y=values,
        marker=dict(
            color=values,
            colorscale='Blues',
            line=dict(color='white', width=2)
        ),
        text=[f'{v:.2%}' for v in values],
        textposition='auto',
        textfont=dict(size=14, color='white', family='Arial Black')
    ))
    
    fig.update_layout(
        title={
            'text': 'Model Performance Metrics',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1e3a8a'}
        },
        xaxis_title='Metric',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    
    return fig

def create_confusion_matrix(tp, tn, fp, fn):
    """
    Create a confusion matrix visualization
    
    Args:
        tp: True Positives
        tn: True Negatives
        fp: False Positives
        fn: False Negatives
        
    Returns:
        plotly figure
    """
    z = [[tn, fp], [fn, tp]]
    x = ['Predicted Normal', 'Predicted Pneumonia']
    y = ['Actual Normal', 'Actual Pneumonia']
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='Blues',
        text=z,
        texttemplate='%{text}',
        textfont={"size": 20, "color": "white"},
        showscale=True
    ))
    
    fig.update_layout(
        title={
            'text': 'Confusion Matrix',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1e3a8a'}
        },
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_roc_curve(fpr, tpr, auc_score):
    """
    Create ROC curve visualization
    
    Args:
        fpr: False Positive Rate array
        tpr: True Positive Rate array
        auc_score: AUC score
        
    Returns:
        plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color='#667eea', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title={
            'text': 'ROC Curve',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1e3a8a'}
        },
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        height=500,
        showlegend=True,
        legend=dict(x=0.6, y=0.1)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    
    return fig


def _ensure_built(model, sample_input):
    """Make sure model has defined outputs (is built)."""
    import tensorflow as tf
    try:
        _ = model.output
    except:
        _ = model(sample_input)  # call once with real-shaped data

def _find_last_conv_layer(model):
    """Find the last Conv2D layer in the model."""
    import tensorflow as tf
    last = None
    for lyr in model.layers:
        if isinstance(lyr, tf.keras.layers.Conv2D):
            last = lyr
    if last is None:
        raise ValueError("No Conv2D layer found for Grad-CAM.")
    return last

def generate_gradcam(model, processed_image, target_is_pneumonia=True, eps=1e-8):
    """
    Generate a Grad-CAM heatmap for a given model and preprocessed image.

    Args:
        model: A compiled Keras/TensorFlow model
        processed_image: Numpy array with shape (1, H, W, C) already preprocessed
        target_is_pneumonia: If True, compute heatmap for the PNEUMONIA prediction signal.
                            Note: model returns probability for NORMAL class, so for
                            pneumonia we use (1 - pred).
        eps: small epsilon to avoid divide-by-zero

    Returns:
        heatmap: 2D numpy array normalized to [0, 1]
        overlay_rgb: RGB numpy array overlaying heatmap on the original image (0-255 uint8)
    """
    import tensorflow as tf
    import numpy as np
    import cv2
    from tensorflow.keras.models import Model

    # Ensure tensor
    if not isinstance(processed_image, tf.Tensor):
        img_batch = tf.convert_to_tensor(processed_image, dtype=tf.float32)
    else:
        img_batch = processed_image

    # Build the model if needed using the real input shape
    _ensure_built(model, img_batch)

    # Find index of last conv layer
    last_conv_idx = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_idx = i
    
    if last_conv_idx is None:
        raise ValueError("No Conv2D layer found in model")
    
    print(f"✓ Using convolutional layer: {model.layers[last_conv_idx].name} (index {last_conv_idx})")
    
    # Create two models:
    # 1. Input -> Conv layer output
    conv_model = Model(model.inputs, model.layers[last_conv_idx].output)
    
    # 2. Conv output -> Final prediction
    # We need to apply layers from last_conv_idx+1 to end
    def apply_remaining_layers(conv_output):
        x = conv_output
        for i in range(last_conv_idx + 1, len(model.layers)):
            x = model.layers[i](x)
        return x
    
    with tf.GradientTape() as tape:
        # Get conv features  
        conv_out = conv_model(img_batch)
        
        # Watch the conv output
        tape.watch(conv_out)
        
        # Apply remaining layers
        preds = apply_remaining_layers(conv_out)
        
        print(f"✓ Forward pass complete")
        print(f"  Conv output shape: {conv_out.shape}")
        print(f"  Predictions shape: {preds.shape}")
        print(f"  Prediction value: {preds[0, 0].numpy():.4f}")

        # For binary sigmoid output
        if preds.shape[-1] == 1:
            class_channel = preds[:, 0]
            # If target is pneumonia, use (1 - normal_prob)
            if target_is_pneumonia:
                class_channel = 1.0 - class_channel
        else:
            # Multiclass: use argmax
            class_idx = tf.argmax(preds[0])
            class_channel = preds[:, class_idx]
        
        print(f"  Class channel value: {float(class_channel.numpy()):.4f}")

    # Gradients wrt conv features
    print(f"✓ Computing gradients...")
    grads = tape.gradient(class_channel, conv_out)
    print(f"  Gradients type: {type(grads)}")
    print(f"  Gradients is None: {grads is None}")
    
    if grads is None:
        raise ValueError("Gradients are None. Cannot compute Grad-CAM.")
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)

    # Normalize to [0,1]
    maxv = tf.reduce_max(heatmap)
    heatmap = heatmap / (maxv + eps)
    heatmap = heatmap.numpy()

    # Resize heatmap to input image size
    H, W = img_batch.shape[1], img_batch.shape[2]
    heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

    # Prepare original image for overlay (uint8, 3-channel)
    img_np = img_batch[0].numpy()
    # If input was 1-channel, repeat to 3 for visualization
    if img_np.shape[-1] == 1:
        orig_rgb = np.repeat((img_np * 255.0).astype(np.uint8), 3, axis=-1)
    else:
        # assume already 0-1 float
        orig_rgb = (img_np * 255.0).astype(np.uint8)

    # Colorize heatmap and blend
    hm_uint8 = np.uint8(255 * heatmap_resized)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR), 1.0,
                                  hm_color, 0.4, 0)

    return heatmap_resized, overlay_bgr

