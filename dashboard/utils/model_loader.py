import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, InceptionV3, MobileNetV2
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_eff, decode_predictions as decode_eff
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inc, decode_predictions as decode_inc
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mob, decode_predictions as decode_mob

@st.cache_resource(show_spinner="Loading Deep Learning Models into cache...")
def load_model_config(model_name):
    """
    Loads and caches the specified model and its preprocessing functions.
    This prevents the app from reloading massive models on every user interaction.
    """
    if model_name == 'EfficientNetB0':
        model = EfficientNetB0(weights='imagenet')
        return {
            'model': model,
            'target_size': (224, 224),
            'preprocess_fn': preprocess_eff,
            'decode_fn': decode_eff,
            'clip_min': 0.0, 'clip_max': 255.0, 'eps_scale': 127.5
        }
    elif model_name == 'InceptionV3':
        model = InceptionV3(weights='imagenet')
        return {
            'model': model,
            'target_size': (299, 299),
            'preprocess_fn': preprocess_inc,
            'decode_fn': decode_inc,
            'clip_min': -1.0, 'clip_max': 1.0, 'eps_scale': 1.0
        }
    elif model_name == 'MobileNetV2':
        model = MobileNetV2(weights='imagenet')
        return {
            'model': model,
            'target_size': (224, 224),
            'preprocess_fn': preprocess_mob,
            'decode_fn': decode_mob,
            'clip_min': -1.0, 'clip_max': 1.0, 'eps_scale': 1.0
        }
    else:
        raise ValueError("Unknown model name.")