# pages/2_🩹_Defenses.py

import streamlit as st
import tensorflow as tf
import numpy as np
import time
from PIL import Image
import os

from utils.model_loader import load_model_config
from utils.attacks import fgsm_attack, pgd_attack, square_attack
from utils.defenses import (
    jpeg_defense, bit_depth_reduction, median_smoothing, feature_squeeze,
    randomized_smoothing_predict, ABSTAIN,
)

st.set_page_config(page_title="Defenses", page_icon="🩹", layout="wide")

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

# Brief, one-paragraph explanation of each defense, shown on the main page once selected.
DEFENSE_INFO = {
    "JPEG Compression": "(Dziugaite et al., 2016). Re-encodes the image through lossy JPEG "
    "compression. JPEG's DCT-based quantization discards exactly the kind of high-frequency detail "
    "that gradient-based attacks tend to inject (see `perturbation_analysis/01_fourier_spectral_analysis.ipynb`), "
    "so a moderately aggressive quality setting can wipe out a lot of the perturbation as a side "
    "effect, no knowledge of the attack required.",
    "Bit-Depth Reduction": "(Xu et al., 2017, *feature squeezing*). Rounds each color channel down "
    "from 8 bits to a handful of coarser levels. A small adversarial perturbation is often smaller "
    "than the gap between two reduced-depth buckets, so it gets rounded away entirely, while the "
    "image's real content survives at much lower resolution.",
    "Median Smoothing": "(Xu et al., 2017, *feature squeezing*). A spatial median filter: each pixel "
    "is replaced by the median of its neighborhood. Adversarial noise tends to be high-frequency and "
    "spatially uncorrelated between neighboring pixels, exactly what a median filter is good at "
    "erasing, at the cost of blurring genuine fine detail too.",
    "Feature Squeezing": "(Xu et al., 2017). The combination of the two defenses above, bit-depth "
    "reduction followed by median smoothing, stacking a color-precision attack surface reduction "
    "with a spatial one.",
    "Randomized Smoothing (Certified)": "(Cohen et al., 2019). The only defense here with a formal "
    "guarantee, not just an empirical one: classifies many noisy copies of the image and returns the "
    "majority vote, together with a **certified L2 radius**, a provable statement that no perturbation "
    "smaller than that radius can change the prediction, regardless of how it's crafted.",
}

st.sidebar.title("⚙️ Defense Setup")

st.sidebar.markdown("**1. Model**")
selected_model = st.sidebar.selectbox("Select CNN Architecture", ['MobileNetV2', 'EfficientNetB0', 'InceptionV3', 'TrafficNet (GTSRB)'])

st.sidebar.markdown("**2. Attack to Defend Against**")
selected_attack = st.sidebar.selectbox(
    "Source attack",
    ['FGSM', 'PGD', 'Square Attack'],
    help="Generates the adversarial image the defense below will be tested against. Kept to a "
         "compact list of representative white-box (FGSM/PGD) and black-box (Square Attack) attacks, "
         "the full 8-attack list lives on the Live Attacks page.",
)
base_eps = st.sidebar.slider("Perturbation Magnitude (ε)", min_value=0.001, max_value=0.1, value=0.03, step=0.005)
if selected_attack == 'PGD':
    attack_iters = st.sidebar.slider("Iterations", min_value=5, max_value=50, value=10, step=5)
if selected_attack == 'Square Attack':
    attack_query_budget = st.sidebar.slider("Query Budget", min_value=50, max_value=1000, value=300, step=50)

st.sidebar.markdown("---")
st.sidebar.markdown("**3. Defense**")
selected_defense = st.sidebar.selectbox(
    "Apply to the adversarial image",
    ['JPEG Compression', 'Bit-Depth Reduction', 'Median Smoothing', 'Feature Squeezing', 'Randomized Smoothing (Certified)'],
)
if selected_defense == 'JPEG Compression':
    jpeg_quality = st.sidebar.slider("JPEG Quality", min_value=10, max_value=95, value=75, step=5, help="Lower quality discards more high-frequency detail, where adversarial noise tends to live.")
if selected_defense == 'Bit-Depth Reduction':
    bit_depth = st.sidebar.slider("Bits per Channel", min_value=1, max_value=7, value=4, step=1, help="Reduces color depth from 8 bits, collapsing small adversarial perturbations back to the same bucket.")
if selected_defense == 'Median Smoothing':
    median_size = st.sidebar.slider("Filter Size", min_value=2, max_value=7, value=3, step=1, help="Spatial median filter window; larger windows smooth out more noise but blur more detail.")
if selected_defense == 'Feature Squeezing':
    fs_bits = st.sidebar.slider("Bits per Channel", min_value=1, max_value=7, value=4, step=1, key="fs_bits")
    fs_size = st.sidebar.slider("Filter Size", min_value=2, max_value=7, value=3, step=1, key="fs_size")
if selected_defense == 'Randomized Smoothing (Certified)':
    rs_sigma = st.sidebar.slider("Noise Level (σ)", min_value=0.05, max_value=0.5, value=0.15, step=0.05, help="Larger σ gives a bigger certified radius but costs more clean accuracy.")
    st.sidebar.caption("Runs 350 noisy forward passes to certify a robustness radius, a few seconds slower than the other defenses.")

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
execute_btn = st.sidebar.button("🩹 Attack, Then Defend", use_container_width=True, type="primary")

st.title("🩹 Defenses")
st.markdown(
    "Generates an adversarial image with the chosen attack, then applies a defense to it and "
    "re-classifies, so you can see live whether the defense recovers the original prediction. "
    "Head back to **⚔️ Live Attacks** to explore the attacks themselves in more depth."
)

st.info(f"**{selected_defense}**  \n{DEFENSE_INFO[selected_defense]}")

if uploaded_file is None:
    st.info("👈 Please upload an image from the sidebar to start.")
else:
    image_pil = Image.open(uploaded_file).convert('RGB')
    config = load_model_config(selected_model)
    model = config['model']

    input_tensor = preprocess_for_model(image_pil, config['target_size'], config['preprocess_fn'])
    orig_preds = model.predict(input_tensor, verbose=0)
    decoded_orig = config['decode_fn'](orig_preds, top=1)[0][0]
    orig_class_idx = int(np.argmax(orig_preds[0]))
    orig_label_tensor = tf.reshape(tf.one_hot(orig_class_idx, orig_preds.shape[-1]), (1, -1))

    if not execute_btn:
        st.image(image_pil, caption=f"Original Image — predicted {decoded_orig[1].capitalize()} ({decoded_orig[2]*100:.1f}%)", width=420)
        st.info("Click **Attack, Then Defend** in the sidebar to run the pipeline.")
    else:
        with st.spinner(f"Generating {selected_attack} attack..."):
            eps_scaled = base_eps * config['eps_scale']
            if selected_attack == 'FGSM':
                adv_tensor = fgsm_attack(input_tensor, orig_label_tensor, eps_scaled, model, config['clip_min'], config['clip_max'])
            elif selected_attack == 'PGD':
                adv_tensor = pgd_attack(input_tensor, orig_label_tensor, eps_scaled, model, config['clip_min'], config['clip_max'], iters=attack_iters)
            else:  # Square Attack
                adv_tensor, _ = square_attack(input_tensor, orig_class_idx, model, eps_scaled, config['clip_min'], config['clip_max'], query_budget=attack_query_budget)

            adv_preds = model.predict(adv_tensor, verbose=0)
            decoded_adv = config['decode_fn'](adv_preds, top=1)[0][0]
            adv_class_idx = int(np.argmax(adv_preds[0]))
            adv_display_img = deprocess_for_display(adv_tensor, config['clip_min'])
            adv_display_img_resized = np.array(Image.fromarray(adv_display_img).resize(image_pil.size, Image.BILINEAR))

        fooled = adv_class_idx != orig_class_idx
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_pil, caption=f"Original — {decoded_orig[1].capitalize()} ({decoded_orig[2]*100:.1f}%)", use_container_width=True)
        with col2:
            st.image(adv_display_img_resized, caption=f"Adversarial ({selected_attack}) — {decoded_adv[1].capitalize()} ({decoded_adv[2]*100:.1f}%)", use_container_width=True)
        if fooled:
            st.error(f"🛡️ The attack fooled the model: {decoded_orig[1].capitalize()} → {decoded_adv[1].capitalize()}. Applying the defense below...")
        else:
            st.success("✅ The attack didn't fool the model at this ε. Applying the defense anyway, for reference.")

        st.markdown("---")
        st.markdown(f"### Defense: {selected_defense}")

        with st.spinner(f"Applying {selected_defense}..."):
            defense_start = time.time()

            if selected_defense == 'Randomized Smoothing (Certified)':
                pred_idx, radius = randomized_smoothing_predict(
                    model, adv_tensor, rs_sigma, config['clip_min'], config['clip_max'],
                    noise_scale=config['eps_scale'], n0=50, n=300,
                )
                defense_time = time.time() - defense_start

                if pred_idx == ABSTAIN:
                    st.error(f"🚫 **Abstained.** Not enough statistical confidence to certify any class (Time: {defense_time:.2f}s).")
                else:
                    defended_label = config['decode_fn'](np.eye(orig_preds.shape[-1])[pred_idx:pred_idx + 1], top=1)[0][0][1]
                    recovered = pred_idx == orig_class_idx
                    if recovered:
                        st.success(f"✅ **Recovered the correct class!** Certified as **{defended_label.capitalize()}** with an L2 radius of {radius:.3f} (Time: {defense_time:.2f}s).")
                    else:
                        st.warning(f"⚠️ **Certified a different class:** **{defended_label.capitalize()}**, radius {radius:.3f} (Time: {defense_time:.2f}s). The attack survived certification.")
            else:
                if selected_defense == 'JPEG Compression':
                    defended_uint8 = jpeg_defense(adv_display_img_resized, quality=jpeg_quality)
                elif selected_defense == 'Bit-Depth Reduction':
                    defended_uint8 = bit_depth_reduction(adv_display_img_resized, bits=bit_depth)
                elif selected_defense == 'Median Smoothing':
                    defended_uint8 = median_smoothing(adv_display_img_resized, size=median_size)
                else:  # Feature Squeezing
                    defended_uint8 = feature_squeeze(adv_display_img_resized, bits=fs_bits, size=fs_size)

                defended_pil = Image.fromarray(defended_uint8).resize(config['target_size'])
                defended_tensor = preprocess_for_model(defended_pil, config['target_size'], config['preprocess_fn'])
                defended_preds = model.predict(defended_tensor, verbose=0)
                decoded_defended = config['decode_fn'](defended_preds, top=3)[0]
                defended_class_idx = np.argmax(defended_preds[0])
                defense_time = time.time() - defense_start

                recovered = defended_class_idx == orig_class_idx
                if recovered:
                    st.success(f"✅ **Recovered the correct class!** The defended image is now classified as **{decoded_defended[0][1].capitalize()}** again (Time: {defense_time:.2f}s).")
                else:
                    st.warning(f"⚠️ **Still fooled.** The defense changed the prediction, but not back to the original class (Time: {defense_time:.2f}s).")

                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.image(defended_uint8, caption=f"Adversarial Image After {selected_defense}", use_container_width=True)
                with col_d2:
                    st.markdown("##### Predictions (Defended):")
                    for i, (net_id, label, prob) in enumerate(decoded_defended):
                        color = "red" if i == 0 and label != decoded_orig[1] else "normal"
                        st.progress(float(prob), text=f"🚨 {label.capitalize()} ({prob*100:.1f}%)" if color == "red" else f"{label.capitalize()} ({prob*100:.1f}%)")
