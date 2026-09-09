# utils/explainability.py

import numpy as np
import tensorflow as tf
from matplotlib import cm

# Last convolutional layer to read activations from, per architecture. TrafficNet shares
# MobileNetV2's backbone, so it reuses the same layer name.
GRADCAM_LAYERS = {
    'MobileNetV2': 'out_relu',
    'EfficientNetB0': 'top_activation',
    'InceptionV3': 'mixed10',
    'TrafficNet (GTSRB)': 'out_relu',
}

def gradcam(img, model, layer_name, class_idx):
    """Grad-CAM (Selvaraju et al., 2017) heatmap for `class_idx`, resized to the input's spatial size."""
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img, training=False)
        class_score = preds[:, class_idx]

    grads = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # alpha_k^c

    heatmap = tf.reduce_sum(conv_output[0] * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (img.shape[1], img.shape[2]))
    return heatmap.numpy().squeeze()

def overlay_heatmap(base_image_uint8, heatmap, alpha=0.4):
    """Blends a Grad-CAM heatmap (jet colormap) onto a denormalized uint8 RGB image."""
    heatmap_colored = cm.jet(heatmap)[..., :3] * 255.0
    overlay = (1 - alpha) * base_image_uint8.astype(np.float32) + alpha * heatmap_colored
    return np.clip(overlay, 0, 255).astype(np.uint8)
