# pages/5_🛡️_Report_Card.py

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import glob
import io
import os
import tempfile
import textwrap
import time
from PIL import Image

from utils.model_loader import load_model_config
from utils.attacks import fgsm_attack, pgd_attack, square_attack

# Set page config
st.set_page_config(page_title="Robustness Report Card", page_icon="🛡️", layout="wide")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.dirname(CURRENT_DIR)
REPO_ROOT = os.path.dirname(DASHBOARD_DIR)
CSS_PATH = os.path.join(DASHBOARD_DIR, "assets", "style.css")
SAMPLE_IMAGES_DIR = os.path.join(REPO_ROOT, "images", "miniimagenet_random_100")

# Function to load local CSS for custom styling
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css(CSS_PATH)

def default_preprocess(clip_min, clip_max):
    """Rescales [0, 255] into [clip_min, clip_max], a reasonable default for a model of unknown provenance."""
    def _fn(img_array):
        return clip_min + (img_array / 255.0) * (clip_max - clip_min)
    return _fn

# Shared "Slate + Blue" palette, same tokens as the dashboard's own light theme (assets/style.css)
PDF_ACCENT = "#2563eb"
PDF_TEXT = "#0f172a"
PDF_TEXT_SEC = "#334155"
PDF_TEXT_MUTED = "#64748b"
PDF_BORDER = "#e2e8f0"
PDF_BG_SOFT = "#f1f5f9"
PDF_SUCCESS = "#16a34a"
PDF_WARN = "#d97706"
PDF_FAIL = "#dc2626"


def _asr_color(asr: float) -> str:
    return PDF_FAIL if asr >= 50 else PDF_WARN if asr >= 25 else PDF_SUCCESS


def _pdf_page_chrome(fig, subtitle, page_label):
    """Accent band + header block shared by every page of the report."""
    fig.add_artist(plt.Rectangle((0, 0.988), 1, 0.012, transform=fig.transFigure,
                                  facecolor=PDF_ACCENT, edgecolor="none"))
    fig.text(0.055, 0.965, "ROBUSTNESS REPORT CARD", fontsize=10, fontweight="bold",
              color=PDF_ACCENT, family="monospace")
    fig.text(0.945, 0.965, "adversarial_attacks_vision", fontsize=8, color=PDF_TEXT_MUTED, ha="right")
    fig.text(0.055, 0.925, subtitle, fontsize=9, color=PDF_TEXT_MUTED)
    fig.text(0.945, 0.925, page_label, fontsize=8, color=PDF_TEXT_MUTED, ha="right")
    fig.add_artist(plt.Line2D([0.055, 0.945], [0.912, 0.912], color=PDF_BORDER,
                               linewidth=1, transform=fig.transFigure))


def _pdf_section_label(fig, x, y, text):
    fig.add_artist(plt.Rectangle((x, y - 0.004), 0.012, 0.012, transform=fig.transFigure, facecolor=PDF_ACCENT))
    fig.text(x + 0.02, y, text, fontsize=10, fontweight="bold", color=PDF_TEXT)


def build_pdf_report(model_label, generated_at, num_images, battery, verdict, verdict_color,
                      robustness_score, summary, results_df, base_epsilon, eps_scale, square_budget):
    """Renders a 2-page PDF report card with matplotlib (no extra PDF dependency needed):
    page 1 is an executive summary (KPIs, ASR, perturbation magnitude, results table), page 2 is a
    per-image x attack outcome grid plus the battery configuration used to generate it."""
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # ==================== PAGE 1: EXECUTIVE SUMMARY ====================
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        _pdf_page_chrome(fig, f"Generated {generated_at}  ·  {num_images} images  ·  {len(battery)} attacks",
                          "Page 1/2 — Executive Summary")
        fig.text(0.055, 0.885, model_label, fontsize=24, fontweight="bold", color=PDF_TEXT)

        # --- KPI stat cards ---
        weakest_attack = summary["ASR"].idxmax()
        weakest_asr = summary["ASR"].max()
        stealthiest_attack = summary["Avg_L2"].idxmin()
        kpi_cards = [
            ("ROBUSTNESS SCORE", f"{robustness_score:.0f} / 100", verdict_color),
            ("VERDICT", verdict, verdict_color),
            ("WEAKEST ATTACK", f"{weakest_attack}  ({weakest_asr:.0f}%)", PDF_FAIL),
            ("STEALTHIEST ATTACK", stealthiest_attack, PDF_TEXT_SEC),
        ]
        card_w, gap, start_x, card_top, card_h = 0.208, 0.018, 0.055, 0.85, 0.075
        for i, (label, value, color) in enumerate(kpi_cards):
            x = start_x + i * (card_w + gap)
            ax_card = fig.add_axes((x, card_top - card_h, card_w, card_h))
            ax_card.axis("off")
            ax_card.set_xlim(0, 1)
            ax_card.set_ylim(0, 1)
            ax_card.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=PDF_BG_SOFT, edgecolor=PDF_BORDER, linewidth=1))
            ax_card.text(0.1, 0.7, label, fontsize=6.6, color=PDF_TEXT_MUTED, fontweight="bold")
            ax_card.text(0.1, 0.3, value, fontsize=11.5, color=color, fontweight="bold", va="center")

        # --- Score donut + quick facts ---
        row1_top, row1_h = 0.745, 0.15
        _pdf_section_label(fig, 0.055, row1_top, "OVERALL RESULT")
        ax_donut = fig.add_axes((0.09, row1_top - row1_h, 0.24, row1_h - 0.01))
        ax_donut.pie(
            [robustness_score, max(100 - robustness_score, 0.001)],
            colors=[verdict_color, PDF_BORDER], startangle=90, counterclock=False,
            wedgeprops=dict(width=0.34),
        )
        ax_donut.text(0, 0.12, f"{robustness_score:.0f}", fontsize=28, fontweight="bold", ha="center", va="center", color=PDF_TEXT)
        ax_donut.text(0, -0.22, "/ 100", fontsize=10, ha="center", va="center", color=PDF_TEXT_MUTED)

        ax_facts = fig.add_axes((0.40, row1_top - row1_h, 0.53, row1_h - 0.01))
        ax_facts.axis("off")
        ax_facts.set_xlim(0, 1)
        ax_facts.set_ylim(0, 1)
        ax_facts.add_patch(plt.Rectangle((0.05, 0.62), 0.22, 0.28, facecolor=verdict_color))
        ax_facts.text(0.16, 0.76, verdict, ha="center", va="center", fontsize=12, fontweight="bold", color="white")
        fact_lines = [
            f"Battery: {', '.join(a for a, _ in battery)}",
            f"Perturbation budget: ε = {base_epsilon:.2f}  (× {eps_scale:g} for this model's input range)",
            f"Square Attack query budget: {square_budget}" if any(a == "Square Attack" for a, _ in battery) else None,
            f"Score is 100 − mean(ASR) across all {len(battery)} attacks in the battery, on {num_images} images.",
        ]
        y = 0.4
        for line in fact_lines:
            if line is None:
                continue
            for wrapped_line in textwrap.wrap(line, width=70) or [""]:
                ax_facts.text(0.05, y, wrapped_line, fontsize=8.6, color=PDF_TEXT_SEC)
                y -= 0.1

        # --- Attack Success Rate bar chart ---
        row2_top = 0.565
        _pdf_section_label(fig, 0.055, row2_top, "ATTACK SUCCESS RATE")
        ax_bar = fig.add_axes((0.30, 0.435, 0.62, 0.11))
        bar_colors = [_asr_color(a) for a in summary["ASR"]]
        bars = ax_bar.barh(summary.index, summary["ASR"], color=bar_colors, height=0.55, zorder=3)
        ax_bar.set_xlim(0, 118)
        ax_bar.tick_params(labelsize=9, colors=PDF_TEXT_SEC, length=0)
        ax_bar.set_xticks([0, 25, 50, 75, 100])
        ax_bar.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=7.5, color=PDF_TEXT_MUTED)
        ax_bar.set_facecolor("white")
        ax_bar.invert_yaxis()
        for spine in ("top", "right", "left", "bottom"):
            ax_bar.spines[spine].set_visible(False)
        ax_bar.grid(axis="x", color=PDF_BORDER, linewidth=0.8, zorder=0)
        for bar, val in zip(bars, summary["ASR"]):
            ax_bar.text(val + 2.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                        va="center", fontsize=9, fontweight="bold", color=PDF_TEXT)

        # --- Perturbation magnitude (stealth) bar chart ---
        row3_top = 0.385
        _pdf_section_label(fig, 0.055, row3_top, "PERTURBATION MAGNITUDE (AVG. L2 DISTANCE)")
        fig.text(0.055, row3_top - 0.02, "Lower = harder to notice with the naked eye, for the same attack success rate.",
                  fontsize=7.5, color=PDF_TEXT_MUTED)
        ax_l2 = fig.add_axes((0.30, 0.26, 0.62, 0.085))
        l2_bars = ax_l2.barh(summary.index, summary["Avg_L2"], color=PDF_ACCENT, height=0.55, alpha=0.85, zorder=3)
        ax_l2.tick_params(labelsize=9, colors=PDF_TEXT_SEC, length=0)
        ax_l2.set_facecolor("white")
        ax_l2.invert_yaxis()
        for spine in ("top", "right", "left", "bottom"):
            ax_l2.spines[spine].set_visible(False)
        ax_l2.grid(axis="x", color=PDF_BORDER, linewidth=0.8, zorder=0)
        max_l2 = max(summary["Avg_L2"].max(), 1e-6)
        ax_l2.set_xlim(0, max_l2 * 1.22)
        for bar, val in zip(l2_bars, summary["Avg_L2"]):
            ax_l2.text(val + max_l2 * 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.2f}",
                       va="center", fontsize=9, fontweight="bold", color=PDF_TEXT)

        # --- Results table ---
        row4_top = 0.225
        _pdf_section_label(fig, 0.055, row4_top, "DETAILED RESULTS")
        ax_table = fig.add_axes((0.055, 0.075, 0.89, row4_top - 0.075 - 0.02))
        ax_table.axis("off")
        col_labels = ["Attack", "ASR", "Avg L2", "Avg Queries", "Avg Gen. Time", "N"]
        cell_text = [
            [attack, f"{row.ASR:.1f}%", f"{row.Avg_L2:.2f}",
             "-" if pd.isna(row.Avg_Queries) else f"{row.Avg_Queries:.0f}",
             f"{row.Avg_Gen_Time_Sec:.2f}s", f"{int(row.N)}"]
            for attack, row in summary.iterrows()
        ]
        table = ax_table.table(cellText=cell_text, colLabels=col_labels, loc="upper center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.1)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor(PDF_BORDER)
            if r == 0:
                cell.set_facecolor(PDF_BG_SOFT)
                cell.set_text_props(fontweight="bold", color=PDF_TEXT_SEC, fontsize=8)
            else:
                cell.set_facecolor("#fafbfc" if r % 2 == 0 else "white")
                cell.set_text_props(color=PDF_TEXT)
                if c == 1:  # ASR column, colored + bold like the dashboard's pills
                    asr_val = float(cell_text[r - 1][1].rstrip("%"))
                    cell.set_text_props(color=_asr_color(asr_val), fontweight="bold")

        fig.text(0.5, 0.03, "Generated with adversarial_attacks_vision  ·  github.com/fragompul/adversarial_attacks_vision",
                  ha="center", fontsize=8, color=PDF_TEXT_MUTED)
        pdf.savefig(fig)
        plt.close(fig)

        # ==================== PAGE 2: PER-IMAGE BREAKDOWN ====================
        fig2 = plt.figure(figsize=(8.27, 11.69))
        fig2.patch.set_facecolor("white")
        _pdf_page_chrome(fig2, f"{model_label}  ·  generated {generated_at}", "Page 2/2 — Detailed Breakdown")

        attacks_order = [a for a, _ in battery]
        images_order = list(dict.fromkeys(results_df["Image"]))
        pivot = results_df.pivot(index="Image", columns="Attack", values="Success").reindex(
            index=images_order, columns=attacks_order
        )

        grid_top = 0.885
        _pdf_section_label(fig2, 0.055, grid_top, "OUTCOME BY IMAGE × ATTACK")
        fig2.text(0.055, grid_top - 0.02,
                   "✗ the model was fooled  ·  ✓ the model resisted this attack on this image",
                   fontsize=7.5, color=PDF_TEXT_MUTED)

        n_images = len(images_order)
        n_attacks = len(attacks_order)
        grid_bottom_floor = 0.33
        row_h = min(0.032, (grid_top - 0.06 - grid_bottom_floor) / max(n_images, 1))
        grid_height = row_h * n_images
        grid_top_axes = grid_top - 0.06
        label_col_w = 0.24
        col_w = (0.89 - label_col_w) / max(n_attacks, 1)

        ax_grid = fig2.add_axes((0.055, grid_top_axes - grid_height, 0.89, grid_height))
        ax_grid.set_xlim(0, 0.89)
        ax_grid.set_ylim(0, grid_height)
        ax_grid.axis("off")

        # Column headers (attack names)
        for j, attack in enumerate(attacks_order):
            cx = label_col_w + j * col_w + col_w / 2
            ax_grid.text(cx, grid_height + row_h * 0.15, attack, ha="center", va="bottom",
                         fontsize=7.5, fontweight="bold", color=PDF_TEXT_SEC, transform=ax_grid.transData)

        for i, img_name in enumerate(images_order):
            y_top = grid_height - i * row_h
            row_center = y_top - row_h / 2
            if i % 2 == 0:
                ax_grid.add_patch(plt.Rectangle((0, y_top - row_h), 0.89, row_h, facecolor=PDF_BG_SOFT, edgecolor="none", zorder=0))
            display_name = img_name if len(img_name) <= 28 else img_name[:25] + "..."
            ax_grid.text(0.01, row_center, display_name, ha="left", va="center", fontsize=7.5, color=PDF_TEXT_SEC)

            for j, attack in enumerate(attacks_order):
                cx = label_col_w + j * col_w + col_w / 2
                val = pivot.loc[img_name, attack] if attack in pivot.columns else None
                if pd.isna(val):
                    symbol, color = "–", PDF_TEXT_MUTED
                elif val:
                    symbol, color = "✗", PDF_FAIL
                else:
                    symbol, color = "✓", PDF_SUCCESS
                ax_grid.text(cx, row_center, symbol, ha="center", va="center", fontsize=10,
                             fontweight="bold", color=color)

        # --- Battery configuration / methodology ---
        method_top = grid_top_axes - grid_height - 0.06
        _pdf_section_label(fig2, 0.055, method_top, "BATTERY CONFIGURATION")
        ax_method = fig2.add_axes((0.055, method_top - 0.22, 0.89, 0.19))
        ax_method.axis("off")
        ax_method.set_xlim(0, 1)
        ax_method.set_ylim(0, 1)
        ax_method.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=PDF_BG_SOFT, edgecolor=PDF_BORDER, linewidth=1))
        method_lines = [
            f"Model under test: {model_label}",
            f"Images evaluated: {num_images}  (see grid above for the exact set)",
            f"Attacks in battery: {', '.join(attacks_order)}",
            f"Perturbation budget: ε = {base_epsilon:.2f}, scaled by ×{eps_scale:g} for this model's input range",
        ]
        if "Square Attack" in attacks_order:
            method_lines.append(f"Square Attack query budget: {square_budget} queries/image")
        method_lines.append(
            "Success is measured as \"predicted class changed vs. the model's own clean prediction\" "
            "(robust accuracy convention), no independent ground-truth labels required."
        )
        y = 0.87
        for line in method_lines:
            for wrapped_line in textwrap.wrap(line, width=100) or [""]:
                ax_method.text(0.03, y, wrapped_line, fontsize=8.6, color=PDF_TEXT_SEC)
                y -= 0.1

        fig2.text(0.5, 0.03, "Generated with adversarial_attacks_vision  ·  github.com/fragompul/adversarial_attacks_vision",
                   ha="center", fontsize=8, color=PDF_TEXT_MUTED)
        pdf.savefig(fig2)
        plt.close(fig2)

    buf.seek(0)
    return buf.getvalue()


def load_and_preprocess(image_pil, target_size, preprocess_fn):
    img = image_pil.convert("RGB").resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.cast(img_array, tf.float32)
    img_array = preprocess_fn(img_array)
    return tf.expand_dims(img_array, axis=0)

@st.cache_resource(show_spinner="Loading your uploaded model...")
def load_uploaded_model(file_bytes, suffix):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    model = tf.keras.models.load_model(tmp_path)
    os.unlink(tmp_path)
    return model

# Main UI
st.title("🛡️ Robustness Report Card")
st.markdown("""
Point this at any model already in this repo, or **upload your own**, and get an automated robustness
score: a battery of attacks (white-box FGSM/PGD and black-box Square Attack, which never touches
gradients) runs against a batch of images with no manual tuning.
""")

# Sidebar - Model
st.sidebar.title("⚙️ Report Card Setup")
st.sidebar.subheader("1. Model")

model_source = st.sidebar.radio("Model source", ["Use a model from this repo", "Upload your own (.h5 / .keras)"])

model = None
target_size, clip_min, clip_max, eps_scale, preprocess_fn = None, None, None, None, None

if model_source == "Use a model from this repo":
    selected_model = st.sidebar.selectbox(
        "Model", ["MobileNetV2", "EfficientNetB0", "InceptionV3", "TrafficNet (GTSRB)"]
    )
    config = load_model_config(selected_model)
    model = config["model"]
    target_size = config["target_size"]
    clip_min, clip_max, eps_scale = config["clip_min"], config["clip_max"], config["eps_scale"]
    preprocess_fn = config["preprocess_fn"]
else:
    uploaded_model_file = st.sidebar.file_uploader("Model file", type=["h5", "keras"])
    col_h, col_w = st.sidebar.columns(2)
    target_h = col_h.number_input("Input height", value=224, min_value=32, step=1)
    target_w = col_w.number_input("Input width", value=224, min_value=32, step=1)
    col_min, col_max = st.sidebar.columns(2)
    clip_min = col_min.number_input("clip_min", value=0.0)
    clip_max = col_max.number_input("clip_max", value=1.0)
    eps_scale = st.sidebar.number_input(
        "eps_scale", value=1.0, min_value=0.01,
        help="Multiplies epsilon before attacking, use ~(clip_max - clip_min) / 2 to make epsilon "
        "comparable to the other models in this repo, which all normalize to that convention.",
    )
    target_size = (int(target_h), int(target_w))
    preprocess_fn = default_preprocess(clip_min, clip_max)
    if uploaded_model_file is not None:
        model = load_uploaded_model(uploaded_model_file.getvalue(), os.path.splitext(uploaded_model_file.name)[1])
        st.sidebar.success(f"Loaded: input {model.input_shape}, output {model.output_shape}")

# Sidebar - Images
st.sidebar.subheader("2. Images")
image_source = st.sidebar.radio("Image source", ["Use repo sample images", "Upload my own"])
num_images = st.sidebar.slider("Number of images", 2, 20, 5)

uploaded_images = None
if image_source == "Upload my own":
    uploaded_images = st.sidebar.file_uploader(
        "Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

# Sidebar - Attack battery
st.sidebar.subheader("3. Attack Battery")
run_fgsm = st.sidebar.checkbox("FGSM (white-box)", value=True)
run_pgd = st.sidebar.checkbox("PGD, 10 iterations (white-box)", value=True)
run_square = st.sidebar.checkbox("Square Attack (black-box)", value=True)
base_epsilon = st.sidebar.slider(
    "Epsilon (perceptual, before eps_scale)", 0.01, 0.3, 0.05, step=0.01,
    help="Same convention as the Live Attacks page: multiplied by eps_scale before being applied.",
)
square_budget = st.sidebar.slider("Square Attack query budget", 50, 1000, 300, step=50, disabled=not run_square)

run_btn = st.sidebar.button("🚀 Generate Report Card", use_container_width=True, type="primary")

# Main logic
if not run_btn:
    st.info("👈 Configure a model and image set in the sidebar, then click **Generate Report Card**.")
elif model is None:
    st.warning("⚠️ Please upload a model file first.")
else:
    if image_source == "Use repo sample images":
        image_paths = sorted(glob.glob(os.path.join(SAMPLE_IMAGES_DIR, "*.*")))[:num_images]
        images_pil = [Image.open(p) for p in image_paths]
        image_labels = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    else:
        if not uploaded_images:
            st.warning("⚠️ Please upload at least one image.")
            st.stop()
        images_pil = [Image.open(f) for f in uploaded_images[:num_images]]
        image_labels = [os.path.splitext(f.name)[0] for f in uploaded_images[:num_images]]

    battery = []
    if run_fgsm:
        battery.append(("FGSM", "whitebox"))
    if run_pgd:
        battery.append(("PGD", "whitebox"))
    if run_square:
        battery.append(("Square Attack", "blackbox"))

    if not battery:
        st.warning("⚠️ Select at least one attack in the battery.")
        st.stop()

    rows = []
    progress = st.progress(0.0, text="Running the report card battery...")
    total_steps = len(images_pil) * len(battery)
    step = 0

    for img_idx, img_pil in enumerate(images_pil):
        input_tensor = load_and_preprocess(img_pil, target_size, preprocess_fn)
        orig_preds = model.predict(input_tensor, verbose=0)
        true_idx = int(np.argmax(orig_preds[0]))
        num_classes = orig_preds.shape[-1]
        label_tensor = tf.reshape(tf.one_hot(true_idx, num_classes), (1, -1))
        eps = base_epsilon * eps_scale

        for attack_name, kind in battery:
            start = time.time()
            if attack_name == "FGSM":
                adv = fgsm_attack(input_tensor, label_tensor, eps, model, clip_min, clip_max)
                queries = np.nan
            elif attack_name == "PGD":
                adv = pgd_attack(input_tensor, label_tensor, eps, model, clip_min, clip_max, iters=10)
                queries = np.nan
            else:  # Square Attack
                adv, queries = square_attack(input_tensor, true_idx, model, eps, clip_min, clip_max, query_budget=square_budget)
            gen_time = time.time() - start

            adv_idx = int(np.argmax(model.predict(adv, verbose=0)[0]))
            l2 = float(np.linalg.norm(input_tensor.numpy() - adv.numpy()))

            rows.append({
                "Image": image_labels[img_idx], "Attack": attack_name, "Success": adv_idx != true_idx,
                "L2_Distance": l2, "Queries": queries, "Gen_Time_Sec": gen_time,
            })
            step += 1
            progress.progress(step / total_steps, text=f"Running the report card battery... ({step}/{total_steps})")

    progress.empty()

    results_df = pd.DataFrame(rows)
    summary = results_df.groupby("Attack", sort=False).agg(
        ASR=("Success", "mean"), Avg_L2=("L2_Distance", "mean"),
        Avg_Queries=("Queries", "mean"), Avg_Gen_Time_Sec=("Gen_Time_Sec", "mean"), N=("Success", "count"),
    )
    summary["ASR"] = (summary["ASR"] * 100).round(1)
    robustness_score = float(100 - summary["ASR"].mean())

    st.markdown("---")

    verdict = "PASS" if robustness_score >= 70 else "WARN" if robustness_score >= 40 else "FAIL"
    verdict_color = "#16a34a" if verdict == "PASS" else "#d97706" if verdict == "WARN" else "#dc2626"

    plotly_font = dict(family="-apple-system, Segoe UI, Roboto, sans-serif", color="#334155")

    col_gauge, col_summary = st.columns([1, 2])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=robustness_score,
            title={"text": "Robustness Score", "font": {"size": 16}},
            number={"font": {"size": 40, "color": "#0f172a"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar": {"color": verdict_color},
                "bgcolor": "#f1f5f9",
                "borderwidth": 0,
            },
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="white", font=plotly_font)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_summary:
        st.subheader("Attack Success Rate by Attack")
        fig_bar = go.Figure(go.Bar(
            x=summary["ASR"], y=summary.index, orientation="h",
            marker_color=["#dc2626" if a >= 50 else "#d97706" if a >= 25 else "#16a34a" for a in summary["ASR"]],
            text=[f"{a:.1f}%" for a in summary["ASR"]], textposition="outside",
        ))
        fig_bar.update_layout(
            xaxis_range=[0, 105], xaxis_title="ASR (%)", height=280, margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="white", plot_bgcolor="white", font=plotly_font,
            xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Full Results")
    st.dataframe(summary.style.format({"ASR": "{:.1f}%", "Avg_L2": "{:.2f}", "Avg_Queries": "{:.0f}", "Avg_Gen_Time_Sec": "{:.2f}s"}), use_container_width=True)

    def _asr_row_color(asr: float) -> str:
        return "#dc2626" if asr >= 50 else "#d97706" if asr >= 25 else "#16a34a"

    table_rows_html = "".join(
        f"<tr><td>{attack}</td>"
        f"<td><span class='pill' style='color:{_asr_row_color(row.ASR)};background:{_asr_row_color(row.ASR)}1a;border-color:{_asr_row_color(row.ASR)}4d'>{row.ASR:.1f}%</span></td>"
        f"<td class='mono'>{row.Avg_L2:.2f}</td>"
        f"<td class='mono'>{'-' if pd.isna(row.Avg_Queries) else f'{row.Avg_Queries:.0f}'}</td>"
        f"<td class='mono'>{row.N}</td></tr>"
        for attack, row in summary.iterrows()
    )
    model_label = selected_model if model_source == "Use a model from this repo" else "Uploaded model"
    generated_at = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    html_report = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Robustness Report Card - {model_label}</title>
<style>
  :root {{ --border: #e2e8f0; --text: #0f172a; --text-secondary: #334155; --text-muted: #64748b; --accent: #2563eb; }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif; color: var(--text); background: #f8fafc;
    max-width: 880px; margin: 0 auto; padding: 48px 24px 64px; font-size: 14px; line-height: 1.5;
  }}
  .mono {{ font-family: "SFMono-Regular", ui-monospace, Menlo, monospace; }}
  .badge {{
    display: inline-flex; align-items: center; gap: 6px; background: #dbeafe; color: var(--accent);
    border-radius: 999px; padding: 4px 12px; font-size: 12px; font-weight: 700; margin-bottom: 14px;
  }}
  h1 {{ font-size: 26px; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 6px 0; }}
  .meta {{ color: var(--text-muted); font-size: 13px; margin-bottom: 28px; }}
  .card {{
    background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 24px;
    margin-bottom: 20px; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.06);
  }}
  .summary {{ display: flex; align-items: center; gap: 32px; flex-wrap: wrap; }}
  .verdict-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }}
  .verdict-pill {{
    display: inline-flex; align-items: center; padding: 6px 14px; border-radius: 6px; font-weight: 700;
    font-size: 13px; letter-spacing: 0.03em; color: {verdict_color}; background: {verdict_color}1a;
    border: 1px solid {verdict_color}4d;
  }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 11px 14px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; font-weight: 700; }}
  tr:last-child td {{ border-bottom: none; }}
  .pill {{ display: inline-flex; padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 12px; border: 1px solid; }}
  .footer {{ color: var(--text-muted); font-size: 12px; text-align: center; margin-top: 32px; }}
  .footer a {{ color: var(--accent); text-decoration: none; }}
</style>
</head>
<body>
  <span class="badge">🛡️ Robustness Report Card</span>
  <h1>{model_label}</h1>
  <p class="meta">Generated {generated_at} &middot; {len(images_pil)} images &middot; {len(battery)} attacks</p>

  <div class="card">
    <div class="summary">
      <div style="flex-shrink:0;">{fig_gauge.to_html(include_plotlyjs='cdn', full_html=False, config={'displayModeBar': False})}</div>
      <div>
        <div class="verdict-row">
          <span class="verdict-pill">{verdict}</span>
          <span style="color:var(--text-secondary); font-size:13px;">Averaged across {len(battery)} attacks in the battery</span>
        </div>
      </div>
    </div>
  </div>

  <div class="card">
    {fig_bar.to_html(include_plotlyjs=False, full_html=False, config={'displayModeBar': False})}
  </div>

  <div class="card" style="padding:0; overflow:hidden;">
    <table>
      <tr><th>Attack</th><th>ASR</th><th>Avg L2</th><th>Avg Queries</th><th>N</th></tr>
      {table_rows_html}
    </table>
  </div>

  <p class="footer">Generated with <a href="https://github.com/fragompul/adversarial_attacks_vision">adversarial_attacks_vision</a></p>
</body>
</html>"""

    pdf_bytes = build_pdf_report(
        model_label, generated_at, len(images_pil), battery, verdict, verdict_color, robustness_score, summary,
        results_df, base_epsilon, eps_scale, square_budget,
    )

    col_dl_html, col_dl_pdf = st.columns(2)
    with col_dl_html:
        st.download_button(
            "⬇️ Download Report (HTML)", data=html_report, file_name="robustness_report_card.html",
            mime="text/html", use_container_width=True,
        )
    with col_dl_pdf:
        st.download_button(
            "⬇️ Download Report (PDF)", data=pdf_bytes, file_name="robustness_report_card.pdf",
            mime="application/pdf", use_container_width=True,
        )
