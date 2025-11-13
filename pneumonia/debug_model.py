"""Debug script to understand model structure"""
import sys
sys.path.insert(0, 'utils')
from model_loader import load_model
import tensorflow as tf

# Load model
model = load_model()
print("\n" + "="*60)
print("MODEL STRUCTURE")
print("="*60)
model.summary()

print("\n" + "="*60)
print("LAYER DETAILS")
print("="*60)
for i, layer in enumerate(model.layers):
    print(f"{i:2d}. {layer.name:20s} | {layer.__class__.__name__:15s} | input: {layer.input.shape if hasattr(layer, 'input') else 'N/A'} | output: {layer.output.shape}")
    
print("\n" + "="*60)
print("FINDING LAST CONV LAYER")
print("="*60)
last_conv_idx = None
for i, layer in enumerate(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        print(f"Found Conv2D at index {i}: {layer.name}")
        last_conv_idx = i

print(f"\nLast Conv2D layer index: {last_conv_idx}")
print(f"Layers after last conv: {len(model.layers) - last_conv_idx - 1}")
print("\nLayers after conv:")
for i in range(last_conv_idx + 1, len(model.layers)):
    layer = model.layers[i]
    print(f"  {i}. {layer.name} ({layer.__class__.__name__})")
