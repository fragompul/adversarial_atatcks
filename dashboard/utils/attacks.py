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