# pages/6_🔬_Explainability.py

import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

from utils.model_loader import load_model_config
from utils.attacks import fgsm_attack, pgd_attack, deepfool_attack
from utils.explainability import gradcam, overlay_heatmap, GRADCAM_LAYERS

st.set_page_config(page_title="Explainability", page_icon="🔬", layout="wide")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.dirname(CURRENT_DIR)
CSS_PATH = os.path.join(DASHBOARD_DIR, "assets", "style.css")

def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css(CSS_PATH)

def preprocess_for_model(img_pil, target_size, preprocess_fn):
    img = img_pil.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return preprocess_fn(img_array)

def deprocess_for_display(tensor, clip_min):
    img_array = tensor[0].numpy()
    if clip_min == -1.0:
        img_array = (img_array + 1.0) / 2.0
    img_array = np.clip(img_array * 255.0 if np.max(img_array) <= 1.0 else img_array, 0, 255).astype(np.uint8)
    return img_array

# Sidebar
st.sidebar.title("⚙️ Explainability Settings")
selected_model = st.sidebar.selectbox("Select CNN Architecture", ['MobileNetV2', 'EfficientNetB0', 'InceptionV3', 'TrafficNet (GTSRB)'])
selected_attack = st.sidebar.selectbox("Select Attack", ['FGSM', 'PGD', 'DeepFool'], help="Grad-CAM is computed before and after this attack, for the same (original) target class, so any attention shift you see is the attack hijacking the model's evidence, not a change of question.")
st.sidebar.markdown("---")
epsilon, iters = None, None
if selected_attack in ['FGSM', 'PGD']:
    base_eps = st.sidebar.slider("Perturbation Magnitude (ε)", min_value=0.001, max_value=0.1, value=0.05, step=0.005)
if selected_attack == 'PGD':
    iters = st.sidebar.slider("Iterations", min_value=5, max_value=50, value=10, step=5)
if selected_attack == 'DeepFool':
    iters = st.sidebar.slider("Max Iterations", min_value=5, max_value=50, value=20, step=5)
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
run_btn = st.sidebar.button("🔬 Compute Grad-CAM", use_container_width=True, type="primary")

# Main UI
st.title("🔬 Explainability: Where Does the Model Look, Before and After an Attack?")
st.markdown("""
Grad-CAM (Selvaraju et al., 2017) highlights the regions of an image that most influenced a
model's prediction, by tracing the gradient of the predicted class's score back to the last
convolutional layer. This page runs it **twice for the same class**: once on the clean image,
once on the adversarial one, so any difference in the heatmap is the attack rewriting what the
model treats as evidence, not a different question being asked.
""")

if uploaded_file is None:
    st.info("👈 Upload an image from the sidebar to compute Grad-CAM before and after an attack.")
else:
    image_pil = Image.open(uploaded_file).convert('RGB')
    config = load_model_config(selected_model)
    model = config['model']
    layer_name = GRADCAM_LAYERS[selected_model]

    input_tensor = preprocess_for_model(image_pil, config['target_size'], config['preprocess_fn'])
    orig_preds = model.predict(input_tensor, verbose=0)
    orig_class_idx = int(np.argmax(orig_preds[0]))
    decoded_orig = config['decode_fn'](orig_preds, top=1)[0][0]
    orig_label_tensor = tf.reshape(tf.one_hot(orig_class_idx, orig_preds.shape[-1]), (1, -1))

    if not run_btn:
        st.image(image_pil, caption=f"Original Image — predicted {decoded_orig[1].capitalize()} ({decoded_orig[2]*100:.1f}%)", width=420)
        st.info("Click **Compute Grad-CAM** in the sidebar to run the attack and compare attention maps.")
    else:
        with st.spinner(f"Running {selected_attack} and computing Grad-CAM..."):
            eps_scaled = base_eps * config['eps_scale'] if 'base_eps' in locals() else None
            if selected_attack == 'FGSM':
                adv_tensor = fgsm_attack(input_tensor, orig_label_tensor, eps_scaled, model, config['clip_min'], config['clip_max'])
            elif selected_attack == 'PGD':
                adv_tensor = pgd_attack(input_tensor, orig_label_tensor, eps_scaled, model, config['clip_min'], config['clip_max'], iters=iters)
            else:  # DeepFool
                adv_tensor = deepfool_attack(input_tensor, model, config['clip_min'], config['clip_max'], max_iter=iters)

            adv_preds = model.predict(adv_tensor, verbose=0)
            adv_class_idx = int(np.argmax(adv_preds[0]))
            decoded_adv = config['decode_fn'](adv_preds, top=1)[0][0]

            # Grad-CAM for the ORIGINAL class on both images, an apples-to-apples comparison of
            # "how much evidence for the true answer does the model still see"
            heatmap_orig = gradcam(input_tensor, model, layer_name, orig_class_idx)
            heatmap_adv = gradcam(adv_tensor, model, layer_name, orig_class_idx)

            orig_display = deprocess_for_display(input_tensor, config['clip_min'])
            adv_display = deprocess_for_display(adv_tensor, config['clip_min'])

            overlay_orig = overlay_heatmap(orig_display, heatmap_orig)
            overlay_adv = overlay_heatmap(adv_display, heatmap_adv)

        fooled = adv_class_idx != orig_class_idx
        if fooled:
            st.error(f"🛡️ **Prediction flipped:** {decoded_orig[1].capitalize()} → {decoded_adv[1].capitalize()} ({decoded_adv[2]*100:.1f}%). Watch how the highlighted region moves below.")
        else:
            st.success(f"✅ **Prediction held:** still {decoded_adv[1].capitalize()} ({decoded_adv[2]*100:.1f}%) after the attack.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"##### Clean Image — Grad-CAM for *{decoded_orig[1].capitalize()}*")
            st.image(overlay_orig, use_container_width=True)
        with col2:
            st.markdown(f"##### Adversarial Image — Grad-CAM for *{decoded_orig[1].capitalize()}* (same class)")
            st.image(overlay_adv, use_container_width=True)

        st.caption(
            "Both heatmaps are computed for the same class, the image's original true label, so a "
            "heatmap that goes dark or scatters on the right is the attack actively suppressing the "
            "evidence the model used to get the answer right in the first place."
        )
