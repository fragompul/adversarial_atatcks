import streamlit as st
import tensorflow as tf
import os

# Import models
from tensorflow.keras.applications import EfficientNetB0, InceptionV3, MobileNetV2

# Import specific preprocessing and decoding functions
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_eff, decode_predictions as decode_eff
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inc, decode_predictions as decode_inc
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mob, decode_predictions as decode_mob

# Dynamic Path Resolution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.dirname(CURRENT_DIR)

# GTSRB (Traffic Signs) Setup
GTSRB_CLASSES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 
    7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)', 9: 'No passing', 
    11: 'Right-of-way at intersection', 12: 'Priority road', 13: 'Yield', 14: 'Stop', 
    15: 'No vehicles', 17: 'No entry', 18: 'General caution', 25: 'Road work', 
    27: 'Pedestrians', 33: 'Turn right ahead', 35: 'Ahead only'
}

def preprocess_gtsrb(img_array):
    """Preprocesses images for custom GTSRB models (usually scaled to [0, 1])."""
    return img_array / 255.0

def decode_gtsrb(preds, top=3):
    """Decodes predictions into GTSRB traffic sign labels."""
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        res = [('gtsrb', GTSRB_CLASSES.get(i, f'Unknown Sign ({i})'), pred[i]) for i in top_indices]
        results.append(res)
    return results

@st.cache_resource(show_spinner="Loading Deep Learning Models into cache...")
def load_model_config(model_name):
    """
    Loads and caches the specified model and its preprocessing functions.
    This prevents the app from reloading massive models on every user interaction.
    """
    if model_name == 'MobileNetV2':
        model = MobileNetV2(weights='imagenet')
        return {
            'model': model,
            'target_size': (224, 224),
            'preprocess_fn': preprocess_mob,
            'decode_fn': decode_mob,
            'clip_min': -1.0,
            'clip_max': 1.0,
            'eps_scale': 1.0
        }
    
    elif model_name == 'EfficientNetB0':
        model = EfficientNetB0(weights='imagenet')
        return {
            'model': model,
            'target_size': (224, 224),
            'preprocess_fn': preprocess_eff,
            'decode_fn': decode_eff,
            'clip_min': 0.0,
            'clip_max': 255.0,
            'eps_scale': 127.5
        }
    
    elif model_name == 'InceptionV3':
        model = InceptionV3(weights='imagenet')
        return {
            'model': model,
            'target_size': (299, 299),
            'preprocess_fn': preprocess_inc,
            'decode_fn': decode_inc,
            'clip_min': -1.0,
            'clip_max': 1.0,
            'eps_scale': 1.0
        }
    
    elif model_name == 'TrafficNet (GTSRB)':
        model_path = os.path.join(DASHBOARD_DIR, 'models', 'traffic_sign_model.h5')
        if not os.path.exists(model_path):
            st.error(f"⚠️ Custom model not found at '{model_path}'. Please upload 'traffic_sign_model.h5'.")
            st.stop()
        
        model = tf.keras.models.load_model(model_path)
        return {
            'model': model,
            'target_size': (224, 224),
            'preprocess_fn': preprocess_gtsrb,
            'decode_fn': decode_gtsrb,
            'clip_min': 0.0,
            'clip_max': 1.0,
            'eps_scale': 1.0
        }
    
    else:
        raise ValueError("Unknown model name.")