# utils/plotting.py

import plotly.graph_objects as go
import plotly.express as px

# Consistent colors
MODEL_COLORS = {
    'EfficientNetB0': '#2ca02c', 
    'InceptionV3': '#1f77b4',    
    'MobileNetV2': '#ff7f0e'     
}

def create_radar_chart(df):
    """Generates an interactive Radar Chart for model robustness."""
    categories = ['Baseline', 'FGSM', 'PGD', 'C&W', 'DeepFool', 'T-IFGSM']
    fig = go.Figure()

    for model_name in df['Model'].unique():
        model_df = df[df['Model'] == model_name]
        
        acc_list = []
        for cat in categories:
            val = model_df[model_df['Attack'] == cat]['Accuracy (%)'].values
            acc_list.append(val[0] if len(val) > 0 else 0.0)
        
        fig.add_trace(go.Scatterpolar(
            r=acc_list,
            theta=categories,
            fill='toself',
            name=model_name,
            line=dict(color=MODEL_COLORS.get(model_name, '#333')),
            hoverinfo="name+r+theta"
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
        showlegend=True,
        margin=dict(t=40, b=40)
    )
    return fig

def create_stealthiness_scatter(df_attacks):
    """Generates a scatter plot mapping L2 distortion vs ASR."""
    fig = px.scatter(
        df_attacks, 
        x="Avg_L2", 
        y="ASR (%)", 
        color="Model", 
        symbol="Attack",
        hover_data=['Model', 'Attack', 'Avg_L2', 'ASR (%)'],
        labels={"Avg_L2": "Average Perceptual Distortion (L2 Norm)", "ASR (%)": "Attack Success Rate (%)"},
        color_discrete_map=MODEL_COLORS,
    )
    
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey')))
    
    median_l2 = df_attacks['Avg_L2'].median() if not df_attacks.empty else 10.0
    fig.add_vrect(x0=0, x1=median_l2, fillcolor="green", opacity=0.05, layer="below", line_width=0,
                  annotation_text="Optimal Stealth Zone", annotation_position="top left")
    return fig