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
    
    df_agg, df_raw = None, None
    if os.path.exists(data_path):
        df_agg = pd.read_csv(data_path)
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        
    return df_agg, df_raw

df, df_raw = load_data()

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🕸️ Radar Chart", 
        "🎯 Stealthiness vs Efficacy", 
        "🖼️ Image Vulnerability", 
        "🔥 Transferability Matrix", 
        "📉 Accuracy vs Perturbation"
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

    # Tab 4: Transferability matriz (heatmap)
    with tab4:
        st.subheader("Adversarial Transferability Matrix")
        st.markdown("""
        **Transferability** is a dangerous property of adversarial examples: an attack generated to fool one specific model (Source) can often fool a completely different, unseen model (Target).
        This heatmap shows the Attack Success Rate (ASR) when transferring FGSM attacks across architectures.
        """)
        
        models = ['MobileNetV2', 'EfficientNetB0', 'InceptionV3']
        transfer_matrix = [
            [100.0, 78.4, 72.2], # Attacks crafted on EfficientNetB0
            [95.7, 100.0, 90.2], # Attacks crafted on InceptionV3
            [77.8, 73.3, 100.0]  # Attacks crafted on MobileNetV2
        ]
        
        fig_heat = px.imshow(
            transfer_matrix,
            labels=dict(x="Target Model (Victim)", y="Source Model (Attack Generator)", color="ASR (%)"),
            x=models,
            y=models,
            text_auto=".1f", # Show numbers with 1 decimal
            aspect="auto",
            color_continuous_scale="Reds"
        )
        
        fig_heat.update_layout(height=500, xaxis_title_font=dict(size=14), yaxis_title_font=dict(size=14))
        st.plotly_chart(fig_heat, use_container_width=True)

    # Tab 5: Accuracy vs Perturbation
    with tab5:
        st.subheader("Accuracy Degradation over Perturbation Strength")
        st.markdown("""
        How much noise is required to break a model? This chart sweeps through different magnitude levels ($\epsilon$) of the FGSM attack. 
        Models with curves that stay higher for longer are intrinsically more robust.
        """)
        
        epsilon_sweep = [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        accuracy_curves = {
            'MobileNetV2': [100.0, 96.0, 89.0, 78.0, 34.0, 17.0, 5.0, 5.0, 3.0, 4.0, 2.0, 1.0, 1.0],
            'EfficientNetB0': [100.0, 100.0, 95.0, 93.0, 67.0, 50.0, 14.0, 11.0, 8.0, 5.0, 4.0, 4.0, 1.0],
            'InceptionV3': [100.0, 98.0, 86.0, 77.0, 41.0, 26.0, 9.0, 9.0, 10.0, 9.0, 8.0, 6.0, 5.0]
        }
        
        fig_line = go.Figure()
        
        for model_name, curve_data in accuracy_curves.items():
            fig_line.add_trace(go.Scatter(
                x=epsilon_sweep, 
                y=curve_data, 
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
