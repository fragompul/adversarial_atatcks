# pages/1_⚔️_Live_Attacks.py

import streamlit as st
import tensorflow as tf
import numpy as np
import time
from PIL import Image
import os

# Import custom modules
from utils.model_loader import load_model_config
from utils.attacks import (
    fgsm_attack, pgd_attack, deepfool_attack, cw_attack, targeted_ifgsm_attack,
    square_attack, nes_attack, boundary_attack,
)

# Set page config
st.set_page_config(page_title="Live Attacks", page_icon="⚔️", layout="wide")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.dirname(CURRENT_DIR)
CSS_PATH = os.path.join(DASHBOARD_DIR, "assets", "style.css")

# Function to load local CSS for custom styling
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Call the function to apply styles
load_css(CSS_PATH)

# Helper Functions
def preprocess_for_model(img_pil, target_size, preprocess_fn):
    """Resizes and preprocesses a PIL image for the selected model."""
    img = img_pil.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return preprocess_fn(img_array)

def deprocess_for_display(tensor, clip_min, clip_max):
    """Converts a model tensor back to a displayable [0, 255] format."""
    img_array = tensor[0].numpy()
    if clip_min == -1.0: # Range [-1, 1] to [0, 255]
        img_array = (img_array + 1.0) / 2.0
    # Range [0, 255] or [0, 1] scaled
    img_array = np.clip(img_array * 255.0 if np.max(img_array) <= 1.0 else img_array, 0, 255).astype(np.uint8)
    return img_array

def extract_noise(orig_tensor, adv_tensor, clip_min, multiplier=10):
    """Extracts the difference (noise) and amplifies it for visual inspection."""
    orig_np = deprocess_for_display(orig_tensor, clip_min, None)
    adv_np = deprocess_for_display(adv_tensor, clip_min, None)

    # Calculate absolute difference and amplify
    noise = np.abs(adv_np.astype(np.float32) - orig_np.astype(np.float32))
    noise = np.clip(noise * multiplier, 0, 255).astype(np.uint8)
    return noise

def get_dft_magnitude(img_array):
    """Computes the 2D Discrete Fourier Transform and returns the magnitude spectrum."""
    # Convert to grayscale for frequency analysis
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img_array

    # Perform 2D FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f) # Shift zero-frequency to center

    # Calculate magnitude spectrum with log scale
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

    # Normalize to [0, 255] for display
    norm_magnitude = (magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum) + 1e-8)
    return (norm_magnitude * 255).astype(np.uint8)

# Brief, one-paragraph explanation of each attack, shown on the main page once selected, so the
# choice in the sidebar is never a bare label with no context.
ATTACK_INFO = {
    "FGSM": "**Fast Gradient Sign Method** (Goodfellow et al., 2014). Computes the gradient of the "
    "loss with respect to the input pixels *once*, then takes a single step of size ε in the "
    "direction that increases the loss the fastest. Cheap (one gradient call) but crude, the "
    "perturbation is the same magnitude at every pixel regardless of how much that pixel actually "
    "matters.",
    "PGD": "**Projected Gradient Descent** (Madry et al., 2018). FGSM's iterative refinement: takes "
    "many small gradient-sign steps, projecting back into the ε-ball around the original image after "
    "each one. Consistently stronger than a single FGSM step at the same ε, the standard whitebox "
    "baseline in adversarial ML research.",
    "C&W": "**Carlini & Wagner** (2017). Reformulates the attack as a direct optimization problem: "
    "minimize the perturbation's L2 norm subject to the prediction actually flipping, via gradient "
    "descent on a differentiable surrogate loss. Slower than FGSM/PGD, but typically finds much "
    "smaller, harder-to-detect perturbations for the same success rate.",
    "DeepFool": "(Moosavi-Dezfooli et al., 2016). Iteratively approximates the model's decision "
    "boundary as locally linear and pushes the image just far enough to cross the *nearest* boundary, "
    "estimating the minimal L2 perturbation needed to fool the model, a geometric rather than "
    "optimization-based approach.",
    "Targeted I-FGSM": "**Targeted Iterative FGSM**. Like PGD, but instead of just maximizing the "
    "loss for the true class, it *minimizes* the loss for a class you choose, steering the "
    "misclassification towards a specific, chosen target rather than any wrong answer.",
    "Square Attack": "(Andriushchenko et al., 2020). **Black-box**: never touches a gradient, only "
    "queries the model's output. Randomly perturbs a square patch of pixels each step and keeps the "
    "change only if it makes the true class less likely, a simple, surprisingly effective random "
    "search.",
    "NES": "**Natural Evolution Strategies** (Ilyas et al., 2018). **Black-box**: estimates the "
    "gradient itself by querying the model at many random points around the image and combining the "
    "outputs into a Monte Carlo gradient estimate, then takes a PGD-style step with that estimate "
    "instead of a real one. Every gradient step costs many queries, the most query-expensive attack "
    "here.",
    "Boundary Attack": "(Brendel et al., 2018). **Black-box, decision-based**: the most realistic "
    "and most restrictive threat model, it never sees a score or a gradient, only the model's top-1 "
    "predicted label. Starts from random noise that's already misclassified and walks along the "
    "decision boundary, gradually shrinking the distance back to the original image.",
}
BLACKBOX_ATTACKS = ('Square Attack', 'NES', 'Boundary Attack')

# Sidebar Configuration
st.sidebar.title("⚙️ Attack Parameters")

# Model Selection
selected_model = st.sidebar.selectbox("Select CNN Architecture", ['MobileNetV2', 'EfficientNetB0', 'InceptionV3', 'TrafficNet (GTSRB)'], help="Choose the underlying neural network. Different architectures exhibit distinct geometric vulnerabilities and baseline accuracies. The TrafficNet is a specialized CNN for autonomous driving scenarios.")

# Attack Selection
selected_attack = st.sidebar.selectbox(
    "Select Attack Algorithm",
    ['FGSM', 'PGD', 'C&W', 'DeepFool', 'Targeted I-FGSM', 'Square Attack', 'NES', 'Boundary Attack'],
    help="Whitebox attacks (FGSM through Targeted I-FGSM) use the model's gradients. Square Attack, "
         "NES, and Boundary Attack are black-box: they only ever query the model's output, exactly "
         "the threat model a real attacker without access to your model's internals would face.",
)

# Dynamic Hyperparameters based on Attack
st.sidebar.markdown("---")
epsilon, iters, c_weight = None, None, None
use_threshold = False
conf_threshold = None
query_budget = None

if selected_attack in ['FGSM', 'PGD', 'Targeted I-FGSM', 'Square Attack', 'NES']:
    base_eps = st.sidebar.slider("Perturbation Magnitude (ε)", min_value=0.001, max_value=0.1, value=0.02, step=0.005, help="Higher epsilon means stronger but more visible attacks.")
if selected_attack in ['PGD', 'Targeted I-FGSM']:
    iters = st.sidebar.slider("Iterations", min_value=5, max_value=50, value=10, step=5, help="Number of gradient steps. More iterations create stronger, more refined attacks but require more computation time.")
if selected_attack == 'C&W':
    c_weight = st.sidebar.slider("Confidence Weight (c)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, help="Balances the trade-off between stealthiness (L2 distortion) and attack success. Higher values prioritize misclassification over invisibility.")
    iters = st.sidebar.slider("Max Iterations", min_value=10, max_value=100, value=40, step=10, help="Maximum number of optimization steps for the C&W algorithm to find the optimal noise.")
if selected_attack == 'DeepFool':
    iters = st.sidebar.slider("Max Iterations", min_value=5, max_value=50, value=20, step=5, help="Maximum steps to push the image iteratively towards the nearest decision boundary in the latent space.")
if selected_attack == 'Square Attack':
    query_budget = st.sidebar.slider("Query Budget", min_value=50, max_value=1000, value=300, step=50, help="Number of model queries the attacker is allowed. Cheapest black-box attack here, one query per candidate patch.")
if selected_attack == 'NES':
    query_budget = st.sidebar.slider("Query Budget", min_value=50, max_value=300, value=150, step=50, help="NES estimates a gradient from many queries per step (10 samples = 20 queries), so it is the most expensive attack per query here; kept to a small budget to stay interactive.")
if selected_attack == 'Boundary Attack':
    query_budget = st.sidebar.slider("Query Budget", min_value=100, max_value=2000, value=800, step=100, help="Decision-based: only sees the top-1 predicted label, never a score or a gradient, the most realistic and most query-hungry black-box threat model.")

if selected_attack == 'Targeted I-FGSM':
    st.sidebar.markdown("**🎯 Targeted Settings**")

    if selected_model == 'TrafficNet (GTSRB)':
        TARGET_CLASSES = {
            "Speed limit (30km/h)": 1, "Speed limit (50km/h)": 2, "Speed limit (120km/h)": 8,
            "Priority road": 12, "Yield": 13, "Stop": 14, "No vehicles": 15, "No entry": 17,
            "General caution": 18, "Pedestrians": 27, "Turn right ahead": 33
        }
        target_name = st.sidebar.selectbox("Select Target Class", list(TARGET_CLASSES.keys()), help="The exact traffic sign you want to force the network to falsely predict.")
        target_class_idx = TARGET_CLASSES[target_name]

    else:
        # Curated list of distinct ImageNet classes to avoid a 1000-item dropdown initially
        BASE_TARGET_CLASSES = {
            "Ladybug": 301,
            "Golden Retriever": 207,
            "Gibbon": 368,
            "Giant Panda": 388,
            "Sports Car": 817,
            "Pizza": 963,
            "Espresso": 967,
            "Volcano": 980,
            "Chihuahua": 151,
            "School Bus": 779
        }

        # Extended list for users who want more options
        EXTENDED_TARGET_CLASSES = {
            **BASE_TARGET_CLASSES,
            "Goldfish": 1,
            "Great White Shark": 2,
            "Hammerhead Shark": 3,
            "Ostrich": 9,
            "Bald Eagle": 22,
            "African Chameleon": 47,
            "Peacock": 84,
            "Macaw": 87,
            "Koala": 105,
            "Jellyfish": 107,
            "Flamingo": 130,
            "Pelican": 144,
            "French Bulldog": 245,
            "Lion": 291,
            "Tiger": 292,
            "Monarch Butterfly": 323,
            "Zebra": 340,
            "Hippopotamus": 344,
            "Macaque": 382,
            "African Elephant": 386,
            "Acoustic Guitar": 403,
            "Airliner": 404,
            "Ambulance": 407,
            "Bicycle": 444,
            "Castle": 483,
            "Cellular Telephone": 487,
            "Coffee Mug": 504,
            "Digital Clock": 530,
            "Electric Guitar": 546,
            "Grand Piano": 579,
            "Laptop": 620,
            "Microphone": 651,
            "Space Shuttle": 812,
            "Submarine": 833,
            "Toilet": 861,
            "Tractor": 866,
            "Ice Cream": 928,
            "Cheeseburger": 933,
            "Broccoli": 937,
            "Strawberry": 949,
            "Banana": 954,
            "Daisy": 985,
            "Toilet Tissue": 999
        }

        # Toggle to show more options without cluttering the UI initially
        show_extended = st.sidebar.checkbox("Show extended class list", help="Expand the dropdown to include more specific ImageNet categories.")

        classes_to_show = EXTENDED_TARGET_CLASSES if show_extended else BASE_TARGET_CLASSES

        target_name = st.sidebar.selectbox("Select Target Class", list(classes_to_show.keys()), help="The exact category you want to force the network to falsely predict (sink class).")
        target_class_idx = classes_to_show[target_name]

    use_threshold = st.sidebar.checkbox("Use confidence threshold", help="Require the attack to push the model's confidence for the target class above a specific percentage.")
    if use_threshold:
        conf_threshold = st.sidebar.slider("Target Confidence Threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.05, help="Desired confidence (between 0 and 1) for the targeted class.")

st.sidebar.markdown("---")

# Image Upload
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], help="Upload your own image to test the network live. It will be automatically resized and preprocessed.")
execute_btn = st.sidebar.button("🚀 Run Attack", use_container_width=True, type="primary")

# Main UI
st.title("⚔️ Live Attacks")
st.markdown("Upload an image, configure your attack parameters in the sidebar, and see how the neural network's perception is manipulated in real-time. Head to **🩹 Defenses** afterwards to see whether the result can be neutralized.")

st.info(f"**{selected_attack}**  \n{ATTACK_INFO[selected_attack]}")

if uploaded_file is not None:
    # Load Image
    image_pil = Image.open(uploaded_file).convert('RGB')

    # Load Model Configuration (Cached)
    config = load_model_config(selected_model)
    model = config['model']

    # Preprocess Image
    input_tensor = preprocess_for_model(image_pil, config['target_size'], config['preprocess_fn'])

    # Get Original Prediction
    orig_preds = model.predict(input_tensor, verbose=0)
    decoded_orig = config['decode_fn'](orig_preds, top=3)[0]
    orig_class_idx = np.argmax(orig_preds[0])
    orig_label_tensor = tf.reshape(tf.one_hot(orig_class_idx, orig_preds.shape[-1]), (1, -1))

    if not execute_btn:
        # Show only original state if button hasn't been pressed
        st.info("Image loaded successfully. Ready to attack!")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_pil, caption="Original Image", use_container_width=True)
        with col2:
            st.subheader("Model Prediction (Clean)")
            for i, (net_id, label, prob) in enumerate(decoded_orig):
                st.progress(float(prob), text=f"**{label.capitalize()}** ({prob*100:.1f}%)")
    else:
        # Execute Attack
        st.markdown("---")
        with st.spinner(f"Generating {selected_attack} attack on {selected_model}..."):
            start_time = time.time()

            eps_scaled = base_eps * config['eps_scale'] if 'base_eps' in locals() else None
            queries_used = None

            # Routing to specific attack functions
            if selected_attack == 'FGSM':
                adv_tensor = fgsm_attack(input_tensor, orig_label_tensor, eps_scaled, model, config['clip_min'], config['clip_max'])
            elif selected_attack == 'PGD':
                adv_tensor = pgd_attack(input_tensor, orig_label_tensor, eps_scaled, model, config['clip_min'], config['clip_max'], iters=iters)
            elif selected_attack == 'DeepFool':
                adv_tensor = deepfool_attack(input_tensor, model, config['clip_min'], config['clip_max'], max_iter=iters)
            elif selected_attack == 'C&W':
                # Safe fallback to avoid KeyError
                box_min = config.get('box_min', config['clip_min'])
                box_max = config.get('box_max', config.get('clip_max', 255.0))
                adv_tensor = cw_attack(input_tensor, orig_label_tensor, model, box_min, box_max, c_weight=c_weight, max_iters=iters)
            elif selected_attack == 'Targeted I-FGSM':
                target_tensor = tf.reshape(tf.one_hot(target_class_idx, orig_preds.shape[-1]), (1, -1))
                adv_tensor = targeted_ifgsm_attack(input_tensor, target_tensor, eps_scaled, model, config['clip_min'], config['clip_max'], iters=iters)
            elif selected_attack == 'Square Attack':
                adv_tensor, queries_used = square_attack(input_tensor, orig_class_idx, model, eps_scaled, config['clip_min'], config['clip_max'], query_budget=query_budget)
            elif selected_attack == 'NES':
                adv_tensor, queries_used = nes_attack(input_tensor, orig_class_idx, model, eps_scaled, config['clip_min'], config['clip_max'], query_budget=query_budget)
            elif selected_attack == 'Boundary Attack':
                adv_tensor, queries_used = boundary_attack(input_tensor, orig_class_idx, model, config['clip_min'], config['clip_max'], query_budget=query_budget)

            exec_time = time.time() - start_time

            # Evaluate Adversarial Image
            adv_preds = model.predict(adv_tensor, verbose=0)
            decoded_adv = config['decode_fn'](adv_preds, top=3)[0]
            adv_class_idx = np.argmax(adv_preds[0])

            # Calculate metrics
            l2_dist = np.linalg.norm(input_tensor.numpy() - adv_tensor.numpy())
            noise_img = extract_noise(input_tensor, adv_tensor, config['clip_min'], multiplier=10)
            adv_display_img = deprocess_for_display(adv_tensor, config['clip_min'], config['clip_max'])

            # Restore original sizes
            orig_size = image_pil.size
            noise_img = np.array(Image.fromarray(noise_img).resize(orig_size, Image.BILINEAR))
            adv_display_img = np.array(Image.fromarray(adv_display_img).resize(orig_size, Image.BILINEAR))

            # Evaluate Attack Success and Stealthiness
            # Thresholds must vary by attack type and model preprocessing scale:
            # 1. FGSM applies uniform noise (visible). C&W optimizes noise into textures (invisible even at higher L2).
            # 2. EfficientNet uses [0, 255] scale (massive raw L2), Inception/MobileNet use [-1, 1] scale (small raw L2).
            if config['clip_min'] == 0.0:
                # Model scale [0, 255]
                stealth_thresholds = {
                    'FGSM': 1500.0, 'PGD': 1500.0, 'Targeted I-FGSM': 1500.0, 'DeepFool': 4000.0, 'C&W': 8000.0,
                    'Square Attack': 1500.0, 'NES': 1500.0, 'Boundary Attack': 6000.0,
                }
            else:
                # Model scale [-1, 1]
                stealth_thresholds = {
                    'FGSM': 10.0, 'PGD': 10.0, 'Targeted I-FGSM': 10.0, 'DeepFool': 40.0, 'C&W': 40.0,
                    'Square Attack': 10.0, 'NES': 10.0, 'Boundary Attack': 40.0,
                }

            L2_THRESHOLD = stealth_thresholds.get(selected_attack, 15.0)

            if selected_attack == 'Targeted I-FGSM':
                if use_threshold:
                    target_prob = adv_preds[0][target_class_idx]
                    prediction_changed = (adv_class_idx == target_class_idx) and (target_prob >= conf_threshold)
                else:
                    prediction_changed = (adv_class_idx == target_class_idx)
            else:
                prediction_changed = (adv_class_idx != orig_class_idx)

            is_stealthy = l2_dist <= L2_THRESHOLD

        # Display Results
        if selected_attack in BLACKBOX_ATTACKS:
            iter_msg = f" using {queries_used} queries"
            fail_iter_msg = f" after {queries_used} queries (budget exhausted)"
        else:
            iter_msg = f" in {iters} iterations" if selected_attack != 'FGSM' else ""
            fail_iter_msg = f" after {iters} iterations" if selected_attack != 'FGSM' else ""

        if prediction_changed and is_stealthy:
            st.success(f"🎯 **Attack Successful & Stealthy!** The model was fooled{iter_msg}. (Time: {exec_time:.2f}s | L2 Distortion: {l2_dist:.2f})")
        elif prediction_changed and not is_stealthy:
            st.warning(f"⚠️ **Attack Succeeded but Detected!** The prediction changed{iter_msg}, but the perturbation is too large and easily visible. (Time: {exec_time:.2f}s | L2 Distortion: {l2_dist:.2f} > {L2_THRESHOLD})")
        else:
            st.error(f"🛡️ **Attack Failed!** The model was highly robust and resisted the perturbation{fail_iter_msg}. (Time: {exec_time:.2f}s | L2 Distortion: {l2_dist:.2f})")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image_pil, caption="Original Image", use_container_width=True)
            st.markdown("##### Original Predictions:")
            for i, (net_id, label, prob) in enumerate(decoded_orig):
                st.progress(float(prob), text=f"{label.capitalize()} ({prob*100:.1f}%)")

        with col2:
            st.image(noise_img, caption="Adversarial Noise (Amplified x10)", use_container_width=True)
            st.markdown("<br><br><p style='text-align: center;'>This subtle noise pushes the image across the decision boundary.</p>", unsafe_allow_html=True)

        with col3:
            st.image(adv_display_img, caption="Adversarial Image", use_container_width=True)
            st.markdown("##### New Predictions (Adversarial):")
            for i, (net_id, label, prob) in enumerate(decoded_adv):
                color = "red" if i==0 and label != decoded_orig[0][1] else "normal"
                st.progress(float(prob), text=f"🚨 {label.capitalize()} ({prob*100:.1f}%)" if color=="red" else f"{label.capitalize()} ({prob*100:.1f}%)")

        # Frequency Domain (DFT)
        st.markdown("---")
        st.markdown("### 🌊 Frequency Domain (Discrete Fourier Transform)")
        st.caption("The DFT magnitude spectrum shows how the adversarial attack introduces high-frequency patterns (spread outwards from the center) that disrupt CNN convolutions.")

        # Calculate DFTs
        orig_array = np.array(image_pil)
        dft_orig = get_dft_magnitude(orig_array)
        dft_noise = get_dft_magnitude(noise_img)
        dft_adv = get_dft_magnitude(adv_display_img)

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.image(dft_orig, caption="DFT of Original Image", use_container_width=True, clamp=True)
        with col_f2:
            st.image(dft_noise, caption="DFT of Adversarial Noise", use_container_width=True, clamp=True)
        with col_f3:
            st.image(dft_adv, caption="DFT of Adversarial Image", use_container_width=True, clamp=True)

else:
    st.info("👈 Please upload an image from the sidebar to start experimenting.")
