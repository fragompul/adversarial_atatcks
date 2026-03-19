# pages/3_🌌_Latent_Space.py

import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import glob
import os
from PIL import Image

# Import custom modules
from utils.model_loader import load_model_config
from utils.attacks import fgsm_attack

# Page Configuration
st.set_page_config(page_title="Latent Space Exploration", page_icon="🌌", layout="wide")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.dirname(CURRENT_DIR)
REPO_DIR = os.path.dirname(DASHBOARD_DIR)
IMAGES_DIR = os.path.join(REPO_DIR, "images", "miniimagenet_random_100")
CSS_PATH = os.path.join(DASHBOARD_DIR, "assets", "style.css")

# Function to load local CSS for custom styling
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Call the function to apply styles
load_css(CSS_PATH)

# Helper Functions
@st.cache_data
def load_sample_images(dataset_path, num_images=15):
    """Loads a subset of images for real-time latent space projection."""
    image_paths = glob.glob(os.path.join(dataset_path, '*.*'))[:num_images]
    return image_paths

def preprocess_for_model(img_path, target_size, preprocess_fn):
    """Loads and preprocesses an image."""
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_raw, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, target_size)
    img = preprocess_fn(img)
    return tf.expand_dims(img, axis=0)

# Main UI
st.title("🌌 Latent Space & Attack Vectors")
st.markdown("""
Neural networks perceive images as points in a highly dimensional **Latent Space**. 
An adversarial attack works by calculating a specific vector that pushes an image's representation out of its original class cluster and into a different one.

Using **Principal Component Analysis (PCA)**, we project these thousands of dimensions down to 2D. 
Here, you can visualize the exact trajectory (Attack Vector) that the adversarial noise forces the image to take.
""")

st.sidebar.header("⚙️ Projection Parameters")

# Sidebar Controls
selected_model = st.sidebar.selectbox("Select CNN Architecture", ['MobileNetV2', 'EfficientNetB0', 'InceptionV3'])
selected_attack = st.sidebar.selectbox("Select Attack Algorithm", ['FGSM']) # Kept to FGSM for real-time speed in 2D projection
epsilon = st.sidebar.slider("Perturbation Magnitude (ε)", min_value=0.01, max_value=0.2, value=0.05, step=0.01)
num_samples = st.sidebar.slider("Number of Images to Project", min_value=5, max_value=30, value=15, step=5)

execute_btn = st.sidebar.button("🌌 Generate Latent Projection", type="primary", use_container_width=True)

if execute_btn:
    # Load Configuration
    config = load_model_config(selected_model)
    model = config['model']
    
    # Create Feature Extractor (Extract from the layer before the final Dense classification layer)
    # Typically, the second to last layer is the Global Average Pooling layer
    layer_name = model.layers[-2].name
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    
    # Load Images
    image_paths = load_sample_images(IMAGES_DIR, num_samples)
    
    if len(image_paths) == 0:
        st.error(f"No images found in '{IMAGES_DIR}'. Please ensure the path is correct.")
    else:
        with st.spinner(f"Extracting features and generating attacks for {len(image_paths)} images..."):
            original_features = []
            adversarial_features = []
            labels_info = []
            
            eps_scaled = epsilon * config['eps_scale']
            loss_object = tf.keras.losses.CategoricalCrossentropy()
            
            for img_path in image_paths:
                # Preprocess
                input_tensor = preprocess_for_model(img_path, config['target_size'], config['preprocess_fn'])
                
                # Get Original Prediction & Features
                orig_preds = model(input_tensor, training=False)
                orig_idx = tf.argmax(orig_preds, axis=-1).numpy()[0]
                decoded_orig = config['decode_fn'](orig_preds.numpy(), top=1)[0][0][1]
                
                orig_feat = feature_extractor(input_tensor, training=False)
                original_features.append(orig_feat.numpy()[0])
                
                # Generate Attack
                orig_label_tensor = tf.reshape(tf.one_hot(orig_idx, orig_preds.shape[-1]), (1, -1))
                adv_tensor = fgsm_attack(input_tensor, orig_label_tensor, eps_scaled, model, config['clip_min'], config['clip_max'])
                
                # Get Adversarial Prediction & Features
                adv_preds = model(adv_tensor, training=False)
                decoded_adv = config['decode_fn'](adv_preds.numpy(), top=1)[0][0][1]
                
                adv_feat = feature_extractor(adv_tensor, training=False)
                adversarial_features.append(adv_feat.numpy()[0])
                
                labels_info.append((decoded_orig, decoded_adv))
                
            # Dimensionality Reduction (PCA)
            all_features = np.vstack((original_features, adversarial_features))
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(all_features)
            
            # Split back
            orig_pca = pca_result[:num_samples]
            adv_pca = pca_result[num_samples:]
            
        # Plotting with Plotly
        st.success("Latent space projection completed successfully!")
        
        fig = go.Figure()
        
        # Draw Attack Vectors (Lines connecting original to adversarial)
        for i in range(num_samples):
            fig.add_trace(go.Scatter(
                x=[orig_pca[i, 0], adv_pca[i, 0]],
                y=[orig_pca[i, 1], adv_pca[i, 1]],
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
        # Draw Original Points
        fig.add_trace(go.Scatter(
            x=orig_pca[:, 0],
            y=orig_pca[:, 1],
            mode='markers+text',
            name='Original Images',
            marker=dict(size=12, color='#1f77b4', line=dict(width=2, color='DarkSlateGrey')),
            text=[f"Orig: {info[0]}" for info in labels_info],
            hoverinfo='text',
            textposition="top center"
        ))
        
        # Draw Adversarial Points
        fig.add_trace(go.Scatter(
            x=adv_pca[:, 0],
            y=adv_pca[:, 1],
            mode='markers',
            name='Adversarial Images',
            marker=dict(size=12, color='#d62728', symbol='x', line=dict(width=2, color='DarkRed')),
            text=[f"Adv: {info[1]}" for info in labels_info],
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title=dict(text=f"Latent Space Topology ({selected_model})", font=dict(size=20)),
            xaxis_title=f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)",
            yaxis_title=f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)",
            template="plotly_white",
            height=700,
            hovermode="closest"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation Expander
        with st.expander("💡 How to interpret this chart"):
            st.markdown("""
            * **Blue Circles:** The image as normally perceived by the network.
            * **Red Crosses:** The hacked version of the image.
            * **Dotted Lines (Vectors):** The mathematical translation applied by the attack. Notice how some vectors shoot entirely out of the main cluster; these are successful attacks pushing the image into an incorrect class territory.
            * Hover over the points to see the predicted class labels change!
            """)
else:
    st.info("👈 Configure your parameters and click 'Generate Latent Projection' to visualize the attack vectors.")
