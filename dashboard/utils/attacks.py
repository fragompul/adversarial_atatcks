# utils/attacks.py

import tensorflow as tf
import numpy as np

loss_object = tf.keras.losses.CategoricalCrossentropy()

def fgsm_attack(img, label, epsilon, model, clip_min, clip_max):
    """Fast Gradient Sign Method"""
    with tf.GradientTape() as tape:
        tape.watch(img)
        pred = model(img, training=False)
        loss = loss_object(label, pred)
    grad = tape.gradient(loss, img)
    return tf.clip_by_value(img + epsilon * tf.sign(grad), clip_min, clip_max)

def pgd_attack(img, label, epsilon, model, clip_min, clip_max, iters=10):
    """Projected Gradient Descent"""
    alpha = epsilon / (iters / 2.0)
    adv_img = tf.identity(img)
    for _ in range(iters):
        with tf.GradientTape() as tape:
            tape.watch(adv_img)
            pred = model(adv_img, training=False)
            loss = loss_object(label, pred)
        grad = tape.gradient(loss, adv_img)
        adv_img = adv_img + alpha * tf.sign(grad)
        perturbation = tf.clip_by_value(adv_img - img, -epsilon, epsilon)
        adv_img = tf.clip_by_value(img + perturbation, clip_min, clip_max)
    return adv_img

def cw_attack(img, label, model, box_min, box_max, c_weight=1.0, max_iters=40, lr=0.05):
    """Carlini & Wagner Attack"""
    modifier = tf.zeros_like(img)
    
    for _ in range(max_iters):
        with tf.GradientTape() as tape:
            tape.watch(modifier)
            adv_norm = 0.5 * (tf.tanh(modifier) + 1.0)
            adv_img = adv_norm * (box_max - box_min) + box_min
            
            l2_loss = tf.reduce_sum(tf.square(adv_img - img))
            preds = model(adv_img, training=False)
            
            real_prob = tf.reduce_sum(label * preds, axis=1)
            other_prob = tf.reduce_max((1.0 - label) * preds, axis=1)
            f_loss = tf.maximum(0.0, real_prob - other_prob)
            total_loss = l2_loss + c_weight * f_loss
            
        grads = tape.gradient(total_loss, modifier)
        modifier = modifier - lr * grads
        
    adv_norm = 0.5 * (tf.tanh(modifier) + 1.0)
    return adv_norm * (box_max - box_min) + box_min

def deepfool_attack(img, model, clip_min, clip_max, num_classes=10, overshoot=0.02, max_iter=20):
    """DeepFool"""
    adv_img = tf.identity(img)
    top_classes = tf.argsort(model(adv_img, training=False)[0], direction='DESCENDING')[:num_classes]
    orig_label = tf.cast(top_classes[0], tf.int32)
    curr_label = orig_label
    iteration = 0
    
    while curr_label == orig_label and iteration < max_iter:
        perturbation = tf.zeros_like(adv_img)
        min_w_norm = float('inf')
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(adv_img)
            preds = model(adv_img, training=False)[0]
            orig_loss = preds[orig_label]
            
            target_losses = []
            for k in range(1, num_classes):
                t_class = tf.cast(top_classes[k], tf.int32)
                target_losses.append(preds[t_class])
                
        orig_grad = tape.gradient(orig_loss, adv_img)
        
        for i in range(num_classes - 1):
            target_grad = tape.gradient(target_losses[i], adv_img)
            w_k = target_grad - orig_grad
            f_k = target_losses[i] - orig_loss
            w_norm = tf.norm(w_k)
            if w_norm == 0: continue
            
            distance = tf.abs(f_k) / (w_norm + 1e-6)
            if distance < min_w_norm:
                min_w_norm = distance
                perturbation = (distance * w_k) / (w_norm + 1e-6)
                
        del tape 
        adv_img = tf.clip_by_value(adv_img + (1 + overshoot) * perturbation, clip_min, clip_max)
        curr_label = tf.cast(tf.argmax(model(adv_img, training=False)[0]), tf.int32)
        iteration += 1
        
    return adv_img

def targeted_ifgsm_attack(img, target_label, epsilon, model, clip_min, clip_max, iters=20):
    """Targeted Iterative FGSM"""
    alpha = epsilon / (iters / 2.0)
    adv_img = tf.identity(img)
    for _ in range(iters):
        with tf.GradientTape() as tape:
            tape.watch(adv_img)
            pred = model(adv_img, training=False)
            loss = loss_object(target_label, pred)
        grad = tape.gradient(loss, adv_img)
        adv_img = adv_img - alpha * tf.sign(grad)
        perturbation = tf.clip_by_value(adv_img - img, -epsilon, epsilon)
        adv_img = tf.clip_by_value(img + perturbation, clip_min, clip_max)
    return adv_img

def _margin_loss(probs, true_idx):
    probs = np.array(probs).flatten()
    true_prob = probs[true_idx]
    other_probs = probs.copy()
    other_probs[true_idx] = -1.0
    return float(true_prob - other_probs.max())

def square_attack(img, true_idx, model, epsilon, clip_min, clip_max, query_budget=300, p_init=0.2):
    """Square Attack (score-based black-box, no gradients). Returns (adv_img, queries_used)."""
    _, h, w, c = img.shape
    init_noise = np.random.choice([-epsilon, epsilon], size=(1, 1, w, c))
    init_noise = np.tile(init_noise, (1, h, 1, 1))
    adv_img = tf.clip_by_value(img + init_noise, clip_min, clip_max)

    best_loss = _margin_loss(model(adv_img, training=False), true_idx)
    queries_used = 1

    for i in range(query_budget - 1):
        if best_loss < 0:
            break
        p = p_init * max(1 - i / query_budget, 0.05)
        side = max(int(round(np.sqrt(p * h * w))), 1)
        row = np.random.randint(0, h - side + 1)
        col = np.random.randint(0, w - side + 1)

        candidate = adv_img.numpy().copy()
        patch_val = np.random.choice([-epsilon, epsilon], size=(1, side, side, c))
        candidate[:, row:row + side, col:col + side, :] = np.clip(
            img.numpy()[:, row:row + side, col:col + side, :] + patch_val, clip_min, clip_max
        )
        candidate = tf.convert_to_tensor(candidate, dtype=tf.float32)

        candidate_loss = _margin_loss(model(candidate, training=False), true_idx)
        queries_used += 1
        if candidate_loss < best_loss:
            adv_img = candidate
            best_loss = candidate_loss

    return adv_img, queries_used

def nes_estimate_gradient(input_image, true_idx, model, sigma, n_samples, clip_min, clip_max):
    """Antithetic NES gradient estimate (Ilyas et al., 2018): every sample costs 2 queries."""
    grad_estimate = tf.zeros_like(input_image)
    n_pairs = n_samples // 2
    for _ in range(n_pairs):
        u = tf.random.normal(shape=input_image.shape)
        img_plus = tf.clip_by_value(input_image + sigma * u, clip_min, clip_max)
        img_minus = tf.clip_by_value(input_image - sigma * u, clip_min, clip_max)
        probs_plus = model(img_plus, training=False).numpy().flatten()
        probs_minus = model(img_minus, training=False).numpy().flatten()
        loss_plus = -np.log(probs_plus[true_idx] + 1e-12)
        loss_minus = -np.log(probs_minus[true_idx] + 1e-12)
        grad_estimate += (loss_plus - loss_minus) * u
    grad_estimate = grad_estimate / (2 * sigma * n_pairs)
    return grad_estimate, 2 * n_pairs

def nes_attack(img, true_idx, model, epsilon, clip_min, clip_max,
               query_budget=200, alpha_fraction=0.2, sigma=0.001, population=10):
    """NES (score-based black-box gradient estimation, Ilyas et al., 2018). Returns (adv_img, queries_used)."""
    alpha = epsilon * alpha_fraction
    adv_img = tf.identity(img)
    queries_used = 0
    while queries_used + population + 1 <= query_budget:
        grad_estimate, q = nes_estimate_gradient(adv_img, true_idx, model, sigma, population, clip_min, clip_max)
        queries_used += q
        adv_img = adv_img + alpha * tf.sign(grad_estimate)
        perturbation = tf.clip_by_value(adv_img - img, -epsilon, epsilon)
        adv_img = tf.clip_by_value(img + perturbation, clip_min, clip_max)
        current_probs = model(adv_img, training=False).numpy().flatten()
        queries_used += 1
        if np.argmax(current_probs) != true_idx:
            break
    return adv_img, queries_used

def boundary_attack(img, true_idx, model, clip_min, clip_max,
                     query_budget=500, spherical_step=0.01, source_step=0.01, max_init_tries=200):
    """Decision-based black-box attack (Brendel et al., 2018): only ever sees the top-1 label,
    never scores or gradients. Returns (adv_img, queries_used)."""
    queries_used = 0
    adv_img = None
    for _ in range(max_init_tries):
        candidate = tf.random.uniform(img.shape, minval=clip_min, maxval=clip_max)
        pred_idx = int(tf.argmax(model(candidate, training=False)[0]).numpy())
        queries_used += 1
        if pred_idx != true_idx:
            adv_img = candidate
            break
    if adv_img is None:
        return img, queries_used

    while queries_used < query_budget:
        diff = img - adv_img
        diff_norm = tf.norm(diff)
        eta = tf.random.normal(img.shape)
        eta = eta - (tf.reduce_sum(eta * diff) / (diff_norm ** 2 + 1e-12)) * diff
        eta = eta / (tf.norm(eta) + 1e-12) * spherical_step * diff_norm

        candidate = tf.clip_by_value(adv_img + eta, clip_min, clip_max)
        pred_idx = int(tf.argmax(model(candidate, training=False)[0]).numpy())
        queries_used += 1

        if pred_idx != true_idx:
            candidate2 = tf.clip_by_value(candidate + source_step * (img - candidate), clip_min, clip_max)
            if queries_used < query_budget:
                pred_idx2 = int(tf.argmax(model(candidate2, training=False)[0]).numpy())
                queries_used += 1
                adv_img = candidate2 if pred_idx2 != true_idx else candidate
            else:
                adv_img = candidate

    return adv_img, queries_used