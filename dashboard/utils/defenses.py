# utils/defenses.py

import io
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.ndimage import median_filter
from scipy.stats import norm, binomtest

def jpeg_defense(img_uint8, quality=75):
    """JPEG compression defense (Dziugaite et al., 2016). Expects/returns a uint8 RGB array."""
    buffer = io.BytesIO()
    Image.fromarray(img_uint8).save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer).convert('RGB'))

def bit_depth_reduction(img_uint8, bits=4):
    """Feature squeezing (Xu et al., 2017): reduces color depth from 8 bits to `bits` per channel."""
    levels = 2 ** bits
    step = 256 // levels
    return ((img_uint8 // step) * step).astype(np.uint8)

def median_smoothing(img_uint8, size=3):
    """Feature squeezing (Xu et al., 2017): spatial median filter, applied per channel."""
    return np.stack([median_filter(img_uint8[:, :, c], size=size) for c in range(img_uint8.shape[2])], axis=-1).astype(np.uint8)

def feature_squeeze(img_uint8, bits=4, size=3):
    """Combined feature squeezing: bit-depth reduction followed by median smoothing."""
    return median_smoothing(bit_depth_reduction(img_uint8, bits=bits), size=size)

ABSTAIN = -1

def _sample_noisy_predictions(model, img, sigma_scaled, n, clip_min, clip_max, batch_size=100):
    """Draws `n` noisy copies of `img`, batched through the model, and returns their predicted class indices."""
    all_preds = []
    remaining = n
    while remaining > 0:
        b = min(batch_size, remaining)
        noise = tf.random.normal(shape=(b,) + img.shape[1:], stddev=sigma_scaled)
        noisy_batch = tf.clip_by_value(img + noise, clip_min, clip_max)
        preds = model(noisy_batch, training=False)
        all_preds.append(tf.argmax(preds, axis=-1).numpy())
        remaining -= b
    return np.concatenate(all_preds)

def randomized_smoothing_predict(model, img, sigma, clip_min, clip_max, noise_scale, n0=100, n=500, alpha=0.001, batch_size=100):
    """Randomized smoothing certification (Cohen et al., 2019). Returns (predicted_class, certified_L2_radius),
    or (ABSTAIN, 0.0) if there isn't enough statistical confidence."""
    sigma_scaled = sigma * noise_scale

    selection_preds = _sample_noisy_predictions(model, img, sigma_scaled, n0, clip_min, clip_max, batch_size)
    c_a_hat = np.bincount(selection_preds).argmax()

    estimation_preds = _sample_noisy_predictions(model, img, sigma_scaled, n, clip_min, clip_max, batch_size)
    count_a = int(np.sum(estimation_preds == c_a_hat))

    p_a_lower = binomtest(count_a, n, p=0.5).proportion_ci(confidence_level=1 - 2 * alpha, method='exact').low
    if p_a_lower <= 0.5:
        return ABSTAIN, 0.0

    return int(c_a_hat), float(sigma * norm.ppf(p_a_lower))
