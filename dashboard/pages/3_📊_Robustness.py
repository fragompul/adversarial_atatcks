# pages/2_📊_Robustness.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from utils.plotting import create_radar_chart, create_stealthiness_scatter

# Page Configuration
st.set_page_config(page_title="Robustness Analytics", page_icon="📊", layout="wide")

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

# Define specific colors for each model
MODEL_COLORS = {
    'MobileNetV2': '#ff7f0e', # Orange
    'EfficientNetB0': '#2ca02c', # Green
    'InceptionV3': '#1f77b4' # Blue
}

ATTACK_MARKERS = {'FGSM': 'circle', 'PGD': 'square', 'C&W': 'diamond', 'DeepFool': 'triangle-up', 'T-IFGSM': 'cross'}

# Data Loading
@st.cache_data
def load_data():
    """Loads aggregated and raw metrics from the data folder."""
    data_path = os.path.join(DASHBOARD_DIR, 'data', 'robustness_metrics.csv')
    raw_path = os.path.join(DASHBOARD_DIR, 'data', 'robustness_metrics_raw.csv')
    corruptions_path = os.path.join(DASHBOARD_DIR, 'data', 'natural_corruptions_robustness.csv')
    transfer_path = os.path.join(DASHBOARD_DIR, 'data', 'transferability_matrix.csv')
    class_pairs_path = os.path.join(DASHBOARD_DIR, 'data', 'transferability_top_class_pairs.csv')
    epsilon_sweep_path = os.path.join(DASHBOARD_DIR, 'data', 'epsilon_sweep.csv')

    df_agg, df_raw, df_corruptions, df_transfer, df_class_pairs, df_epsilon = None, None, None, None, None, None
    if os.path.exists(data_path):
        df_agg = pd.read_csv(data_path)
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
    if os.path.exists(corruptions_path):
        df_corruptions = pd.read_csv(corruptions_path)
    if os.path.exists(transfer_path):
        df_transfer = pd.read_csv(transfer_path)
    if os.path.exists(class_pairs_path):
        df_class_pairs = pd.read_csv(class_pairs_path)
    if os.path.exists(epsilon_sweep_path):
        df_epsilon = pd.read_csv(epsilon_sweep_path)

    return df_agg, df_raw, df_corruptions, df_transfer, df_class_pairs, df_epsilon

df, df_raw, df_corruptions, df_transfer, df_class_pairs, df_epsilon = load_data()

# Main UI
st.title("📊 Quantitative Robustness Analytics")
st.markdown("""
Explore the mass-evaluation results of our Convolutional Neural Networks against various adversarial attacks. 
All metrics presented here are derived from the evaluation of **100 random images from the MiniImageNet dataset**.
""")

if df is None:
    st.error("⚠️ Data not found. Please ensure 'robustness_metrics.csv' is inside the 'data/' folder.")
else:
    # Create Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🕸️ Radar Chart",
        "🎯 Stealthiness vs Efficacy",
        "🖼️ Image Vulnerability",
        "🔥 Transferability Matrix",
        "📉 Accuracy vs Perturbation",
        "🌫️ Natural Corruptions",
    ])
    
    # Tab 1: Radar Chart
    with tab1:
        st.subheader("Model Resilience Comparison")
        st.markdown("This spider chart visualizes the retained accuracy of each model under different attack scenarios. A larger area indicates a more robust architecture.")
        
        fig_radar = create_radar_chart(df)
        st.plotly_chart(fig_radar, use_container_width=True)

    # Tab 2: Stealthiness vs Efficacy
    with tab2:
        st.subheader("Attack Performance Trade-off")
        st.markdown("""
        An optimal adversarial attack maximizes the **Attack Success Rate (ASR)** while minimizing the **perceptual distortion ($L_2$ Norm)**. 
        Hover over the points to see specific algorithm performance.
        """)
        
        df_attacks = df[df['Attack'] != 'Baseline'].copy()
        
        fig_scatter = create_stealthiness_scatter(df_attacks)        
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Tab 3: Image vulnerability profile
    with tab3:
        if df_raw is not None:
            st.subheader("Intrinsic Dataset Vulnerability")
            st.markdown("""
            Are some images naturally harder to hack? This histogram shows the distribution of **Vulnerability Scores** across our 100-image dataset. 
            The score represents how many times an image was successfully hacked out of 15 total attempts (3 models $\\times$ 5 attacks).
            """)
            
            # Calculate Vulnerability Score
            df_raw_attacks = df_raw[df_raw['Attack'] != 'Baseline'].copy()
            image_vuln = df_raw_attacks.groupby('Image_ID')['Is_Success'].sum().reset_index()
            image_vuln.rename(columns={'Is_Success': 'Vulnerability_Score'}, inplace=True)
            
            fig_hist = px.histogram(
                image_vuln, 
                x="Vulnerability_Score", 
                nbins=16, 
                range_x=[-0.5, 15.5],
                labels={"Vulnerability_Score": "Vulnerability Score (0 = Robust, 15 = Extremely Fragile)"},
                color_discrete_sequence=['#9467bd'],
                height=500
            )
            
            fig_hist.update_layout(bargap=0.1, yaxis_title="Number of Images")
            
            # Add vertical lines for insights
            fig_hist.add_vline(x=3.5, line_width=2, line_dash="dash", line_color="green", annotation_text="Highly Robust")
            fig_hist.add_vline(x=11.5, line_width=2, line_dash="dash", line_color="red", annotation_text="Highly Fragile")
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Display Quick Stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Score", f"{image_vuln['Vulnerability_Score'].mean():.1f} / 15")
            col2.metric("Indestructible Images (Score 0)", len(image_vuln[image_vuln['Vulnerability_Score'] == 0]))
            col3.metric("Completely Fragile (Score 15)", len(image_vuln[image_vuln['Vulnerability_Score'] == 15]))
            
        else:
            st.info("💡 Raw image data not available for vulnerability profiling. Please ensure 'robustness_metrics_raw.csv' is in the 'data/' folder.")

    # Tab 4: Transferability matrix (heatmap)
    with tab4:
        st.subheader("Adversarial Transferability Matrix")
        st.markdown("""
        **Transferability** is a dangerous property of adversarial examples: an attack generated to fool one specific model (Source) can often fool a completely different, unseen model (Target).
        This heatmap shows the real Attack Success Rate (ASR) measured when transferring attacks across our 3 architectures, on 30 MiniImageNet images (`robustness_evaluation/07_transferability_heatmap.ipynb`).
        """)

        if df_transfer is None:
            st.info("💡 'transferability_matrix.csv' not found in the 'data/' folder.")
        else:
            transfer_models = sorted(set(df_transfer['Source_Model']) | set(df_transfer['Target_Model']))
            selected_transfer_attack = st.selectbox(
                "Source attack", df_transfer['Attack'].unique(), key="transfer_attack_select"
            )

            pivot_transfer = (
                df_transfer[df_transfer['Attack'] == selected_transfer_attack]
                .pivot(index='Source_Model', columns='Target_Model', values='ASR')
                .reindex(index=transfer_models, columns=transfer_models)
            )

            fig_heat = px.imshow(
                pivot_transfer,
                labels=dict(x="Target Model (Victim)", y="Source Model (Attack Generator)", color="ASR (%)"),
                text_auto=".1f",
                aspect="auto",
                color_continuous_scale="Reds",
                zmin=0, zmax=100,
            )
            fig_heat.update_layout(height=500, xaxis_title_font=dict(size=14), yaxis_title_font=dict(size=14))
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption("Diagonal = white-box success rate on the source model itself. Off-diagonal = the actual black-box transfer ASR.")

            if df_class_pairs is not None:
                st.markdown("##### Which classes does a transferred attack land on?")
                st.markdown("""
                Beyond *whether* an attack transfers, this counts *where* it lands: the (source class → target class)
                pairs that recurred most often among successful transfers. Recurring pairs are local evidence of the
                same "universal sink class" phenomenon explored per-model in **🕳️ Attractors & Loss**
                (`adversarial_attractors/04_sink_class_analysis.ipynb`), this time seen through transferability.
                """)
                pairs_for_attack = df_class_pairs[df_class_pairs['Attack'] == selected_transfer_attack].head(10)
                st.dataframe(
                    pairs_for_attack[['Source_Class', 'Target_Class', 'Count']],
                    use_container_width=True, hide_index=True,
                )

    # Tab 5: Accuracy vs Perturbation
    with tab5:
        st.subheader("Accuracy Degradation over Perturbation Strength")
        st.markdown("""
        How much noise is required to break a model? This chart sweeps through different magnitude levels ($\epsilon$) of the FGSM attack, measured for real on 30 MiniImageNet images (`robustness_evaluation/06_epsilon_sweep.ipynb`).
        Models with curves that stay higher for longer are intrinsically more robust. "Accuracy" here is retained agreement with each model's own clean prediction, not independent ImageNet ground truth, the same convention used across this dashboard.
        """)

        if df_epsilon is None:
            st.info("💡 'epsilon_sweep.csv' not found in the 'data/' folder.")
        else:
            fig_line = go.Figure()

            for model_name in df_epsilon['Model'].unique():
                sub = df_epsilon[df_epsilon['Model'] == model_name].sort_values('Epsilon')
                fig_line.add_trace(go.Scatter(
                    x=sub['Epsilon'],
                    y=sub['Accuracy'],
                    mode='lines+markers',
                    name=model_name,
                    line=dict(width=3, color=MODEL_COLORS.get(model_name, '#333')),
                    marker=dict(size=8)
                ))

            fig_line.update_layout(
                xaxis_title="Adversarial Perturbation Strength (ε)",
                yaxis_title="Retained Accuracy (%)",
                yaxis_range=[-5, 105],
                hovermode="x unified", # Shows all values simultaneously on hover
                height=500
            )

            # Highlight vulnerability zone
            fig_line.add_vrect(x0=0.0001, x1=0.01, fillcolor="red", opacity=0.05, layer="below", line_width=0,
                          annotation_text="Critical Drop Zone", annotation_position="top right")

            st.plotly_chart(fig_line, use_container_width=True)

    # Tab 6: Natural Corruptions (not an attack: fog, blur, noise, compression at 3 severities)
    with tab6:
        st.subheader("Robustness to Natural Corruptions (Not Adversarial)")
        st.markdown("""
        Adversarial robustness and robustness to everyday noise are **different properties**: a
        model can resist a carefully optimized attack yet still flip its prediction on plain fog,
        blur, or JPEG artifacts. This tab evaluates all 3 models against the ImageNet-C corruption
        suite (Hendrycks & Dietterich, 2019), 8 corruption types x 3 severities, on 15 held-out
        images, real execution, no attacker involved.
        """)

        if df_corruptions is None:
            st.info("💡 'natural_corruptions_robustness.csv' not found in the 'data/' folder.")
        else:
            overall = df_corruptions.groupby('Model')['Flipped'].mean().sort_values() * 100
            fig_bar = px.bar(
                overall, x=overall.index, y=overall.values,
                labels={'x': 'Model', 'y': 'Overall Flip Rate (%)'},
                color=overall.index, color_discrete_map=MODEL_COLORS,
                text=[f"{v:.1f}%" for v in overall.values],
            )
            fig_bar.update_layout(showlegend=False, yaxis_range=[0, 100], height=420,
                                   title="Average Prediction Flip Rate Across All 8 Corruptions x 3 Severities")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption("Lower is more robust. Ranking here can, and does, differ from the adversarial ranking above, they are measuring different failure modes.")

            st.markdown("##### Flip Rate by Corruption Type and Severity")
            selected_corr_model = st.selectbox("Model", df_corruptions['Model'].unique(), key="corr_model_select")
            pivot = (
                df_corruptions[df_corruptions['Model'] == selected_corr_model]
                .pivot_table(index='Corruption', columns='Severity', values='Flipped', aggfunc='mean') * 100
            )
            fig_heat = px.imshow(
                pivot, text_auto=".0f", aspect="auto", color_continuous_scale="Oranges", zmin=0, zmax=100,
                labels=dict(x="Severity", y="Corruption", color="Flip Rate (%)"),
            )
            fig_heat.update_layout(height=420)
            st.plotly_chart(fig_heat, use_container_width=True)
