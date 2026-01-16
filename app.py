
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
try:
    from st_annotated_text import annotated_text
except ImportError:
    # Fallback if library fails to load
    def annotated_text(*args):
        for arg in args:
            if isinstance(arg, tuple):
                st.write(f"{arg[0]} ({arg[1]})")
            else:
                st.write(arg)
from streamlit_echarts import st_echarts
import pygwalker as pyg
from hea_prediction import HEAModelPipeline, parse_formula, calculate_features, feature_names, element_data

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="HEA Discovery Engine", layout="wide")

# --- 2. CUSTOM STYLING (PREMIUM UI) ---
st.markdown("""
<style>
    /* White Header & Premium Background */
    header {visibility: hidden;}
    .main {
        background-color: #f8f9fa;
        color: #1e1e1e;
    }
    h1, h2, h3 {
        color: #0d1b2a; 
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        font-weight: 600;
    }
    .stApp > header {
        background-color: transparent;
    }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"] label {
        color: #6c757d; /* Muted gray for label */
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #0d1b2a; /* Dark blue for value */
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1b2a; 
        color: #ffffff;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {
        color: #ffffff;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #a0aec0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 4px;
        padding: 10px 20px;
        font-weight: 500;
        border: 1px solid #e0e0e0;
        color: #333333; /* Default text color */
    }
    .stTabs [aria-selected="true"] {
        background-color: #0d1b2a; /* Dark Blue active */
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. BACKEND LOADING ---
@st.cache_resource
def load_backend():
    pipeline = HEAModelPipeline("MPEA_dataset.csv") 
    pipeline.preprocess()
    pipeline.train_and_validate()
    return pipeline

@st.cache_resource
def calculate_shap_values(_pipeline):
    explainer = shap.TreeExplainer(_pipeline.model)
    X_sample = _pipeline.X_final.sample(n=min(100, len(_pipeline.X_final)), random_state=42)
    shap_values = explainer(X_sample)
    return explainer, shap_values, X_sample

try:
    with st.spinner('Initializing Engine...'):
        pipeline = load_backend()
        explainer, shap_values, X_shap_sample = calculate_shap_values(pipeline)
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()

# --- 4. SIDEBAR INPUTS ---
with st.sidebar:
    st.title("HEA Composer")
    st.markdown("Define your alloy parameters below to predict its mechanical properties.")
    
    st.markdown("### 1. Composition")
    formula = st.text_input("Chemical Formula", value="Al1Co1Cr1Fe1Ni1", help="Enter elements and molar ratios (e.g., Al0.3CoCrFeNi)")
    
    st.markdown("### 2. Processing History")
    temp = st.number_input("Test Temp (°C)", min_value=0.0, max_value=2000.0, value=25.0, step=10.0)
    grain_size = st.number_input("Grain Size (μm)", min_value=0.1, max_value=5000.0, value=50.0)
    process_method = st.selectbox("Method", ["CAST", "ANNEAL", "FORGED", "OTHER"])

    predict_btn = st.button("Predict Yield Strength", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.info("The model uses Physics-Informed ML, combining Valence Electron Concentration (VEC), Lattice Distortion, and Thermodynamics.")

# --- 5. MAIN DASHBOARD LOGIC ---

# Default prediction state
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if predict_btn:
    try:
        # Parding
        comp = parse_formula(formula)
        if not comp:
            st.error("Invalid Formula! format: 'AlCoCrFeNi' or 'Al1Co1Cr1Fe1Ni1'")
        else:
            # Calc Features
            base_feats = calculate_features(comp, temp)
            
            input_df = pd.DataFrame([base_feats], columns=feature_names)
            input_df['Grain_Size'] = grain_size
            
            method_clean = 'CAST' if 'CAST' in process_method else ('ANNEAL' if 'ANNEAL' in process_method else 'OTHER')
            proc_vec = pipeline.enc_proc.transform([[method_clean]])
            proc_df = pd.DataFrame(proc_vec, columns=pipeline.enc_proc.get_feature_names_out(['Proc']))
            
            final_vec = pd.concat([input_df, proc_df], axis=1)
            pred_ys = pipeline.model.predict(final_vec)[0]
            
            st.session_state.prediction = {
                "yield_strength": pred_ys,
                "vec": base_feats[0],
                "delta": base_feats[2],
                "s_mix": base_feats[6],
                "composition": comp
            }
    except Exception as e:
        st.error(f"Prediction Failed: {e}")

# --- 6. MAIN DISPLAY AREA ---
st.title("High-Entropy Alloy Engine")

# A) TOP SECTION: PREDICTIONS & MATERIAL PROPERTIES
if st.session_state.prediction:
    res = st.session_state.prediction
    
    # 1. Main KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Yield Strength", f"{res['yield_strength']:.0f} MPa", delta_color="normal")
    with col2:
        val = res['vec']
        label = "BCC (Hard)" if val < 6.87 else ("FCC (Soft)" if val > 8.0 else "Mixed Phase")
        st.metric("VEC (Phase)", f"{val:.2f}", label)
    with col3:
        st.metric("Lattice Distortion (δ)", f"{res['delta']:.2f} %")
    with col4:
        st.metric("Mixing Entropy (S_mix)", f"{res['s_mix']:.2f} R")

    # 2. visual properties
    st.markdown("### Material Properties Snapshot")
    
    # Use Annotated Text for quick summary
    vec_color = "#faa" if res['vec'] < 6.87 else ("#afa" if res['vec'] > 8.0 else "#ffeeba")
    try:
        annotated_text(
            "The alloy ",
            (f"{formula}", "Formula"),
            " is predicted to exist as a ",
            (f"{label}", "Phase", vec_color),
            " structure with a yield strength of ",
            (f"{res['yield_strength']:.0f} MPa", "Strength", "#8ef"),
            "."
        )
    except:
        st.write(f"The alloy {formula} is predicted to exist as a {label} structure with a yield strength of {res['yield_strength']:.0f} MPa.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Radar Chart for Properties using Echarts
    radar_option = {
        "title": {"text": "Physico-Chemical Footprint"},
        "tooltip": {},
        "radar": {
            "indicator": [
                {"name": "VEC", "max": 12},
                {"name": "Delta (%)", "max": 10},
                {"name": "Entropy", "max": 2},
                {"name": "Melting Pt (Norm)", "max": 3000},
            ]
        },
        "series": [{
            "name": "Alloy Properties",
            "type": "radar",
            "data": [
                {
                    "value": [
                        res['vec'], 
                        res['delta'], 
                        res['s_mix'], 
                        2000 # Placeholder for Tm average
                    ],
                    "name": "Current Alloy"
                }
            ]
        }]
    }
    # col_chart, col_comp = st.columns([1, 1])
    # with col_chart:
        # st_echarts(radar_option, height="300px")
    # with col_comp:
    
    st.markdown("---")

else:
    st.info("Enter alloy parameters in the sidebar to generate a prediction.")

# --- 7. TABS FOR ADVANCED TOOLS ---
t_sim, t_shap, t_data = st.tabs(["Strength Simulator", "Model Intelligence", "Dataset Analytics"])

with t_sim:
    st.subheader("Interactive Alloy Designer")
    st.caption("Fine-tune the composition in real-time to see how properties shift.")
    
    # Only if prediction exists, or default
    if st.session_state.prediction:
        base_comp = st.session_state.prediction['composition']
    else:
        base_comp = {'Al':1, 'Co':1, 'Cr':1, 'Fe':1, 'Ni':1} # default

    # Sliders
    new_comp = {}
    cols = st.columns(len(base_comp))
    for i, (el, val) in enumerate(base_comp.items()):
        with cols[i]:
             new_comp[el] = st.slider(f"{el}", 0.0, 2.0, float(val), 0.1, key=f"sim_{el}")
    
    # Real-time update logic similar to before...
    # (Simplified for brevity, can duplicate logic if needed or user interacts)
    
    # --- PROCESSING PARAMETERS FOR SIMULATION ---
    st.markdown("### Processing Conditions")
    c1, c2, c3 = st.columns(3)
    with c1:
        sim_temp = st.slider("Temperature (°C)", 0.0, 1500.0, 25.0, 10.0, key="sim_temp")
    with c2:
        sim_grain = st.slider("Grain Size (μm)", 0.1, 200.0, 50.0, 5.0, key="sim_grain")
    with c3:
        sim_method = st.selectbox("Processing Method", ["CAST", "ANNEAL", "FORGED", "OTHER"], key="sim_method")

    st.markdown("---")

    # --- REAL-TIME CALCULATION ---
    # Normalize composition
    total_moles = sum(new_comp.values())
    if total_moles > 0:
        comp_normalized = {k: v/total_moles for k,v in new_comp.items()}
        
        # Calculate Physics Features
        base_feats = calculate_features(comp_normalized, float(sim_temp))
        
        # Prepare Input
        input_df = pd.DataFrame([base_feats], columns=feature_names)
        input_df['Grain_Size'] = float(sim_grain)
        
        # Handle One-Hot Encoding
        method_clean = 'CAST' if 'CAST' in sim_method else ('ANNEAL' if 'ANNEAL' in sim_method else 'OTHER')
        proc_vec = pipeline.enc_proc.transform([[method_clean]])
        proc_cols = pipeline.enc_proc.get_feature_names_out(['Proc'])
        proc_df = pd.DataFrame(proc_vec, columns=proc_cols)
        
        final_vec = pd.concat([input_df, proc_df], axis=1)
        
        # Predict
        sim_pred_ys = pipeline.model.predict(final_vec)[0]
        
        # DISPLAY RESULT
        col_res, col_chart = st.columns([1, 2])
        
        with col_res:
            st.markdown("### Simulated Strength")
            st.metric("Yield Strength", f"{sim_pred_ys:.0f} MPa")
            
            # Phase Check
            vec_val = base_feats[0]
            phase_lbl = "BCC (Hard)" if vec_val < 6.87 else ("FCC (Soft)" if vec_val > 8.0 else "Mixed")
            st.metric("Phase Prediction", phase_lbl, f"VEC: {vec_val:.2f}")

        with col_chart:
            # Pie Chart
            fig = px.pie(values=list(new_comp.values()), names=list(new_comp.keys()), title="Simulated Composition")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Total composition must be greater than 0")

with t_shap:
    st.subheader("Model Decision Explanations (SHAP)")
    st.markdown("Uncover the 'black box': See how Physics (ΔR) and Processing (Grain Size) drive strength.")

    # 1. Global Summary
    st.markdown("### 1. Global Feature Importance")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Summary (Beeswarm)**")
        st.caption("Dots identify individual alloys. Color = High/Low value of feature.")
        shap.plots.beeswarm(shap_values, max_display=12, show=False)
        st.pyplot(plt.gcf())
        plt.clf() # Clear current figure
        
    with col2:
        st.markdown("**Mean Impact (Bar)**")
        st.caption("Average absolute impact of each feature on Yield Strength.")
        shap.plots.bar(shap_values, max_display=12, show=False)
        st.pyplot(plt.gcf())
        plt.clf()
    
    st.markdown("---")

    # 2. Key Dependencies
    st.markdown("### 2. Physics & Processing Dependencies")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Lattice Distortion (ΔR) Effect**")
        st.caption("How atomic mismatch affects strength.")
        shap.plots.scatter(shap_values[:, "Delta_R"], color=shap_values, show=False)
        st.pyplot(plt.gcf())
        plt.clf()
        
    with col4:
        st.markdown("**Grain Size Effect**")
        st.caption("Hall-Petch relationship visualization.")
        shap.plots.scatter(shap_values[:, "Grain_Size"], color=shap_values, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

    st.markdown("---")

    # 3. Local Explanation
    st.markdown("### 3. Single Prediction Analysis (Waterfall)")
    st.caption("Select a specific test alloy to see exactly why it got its predicted score.")
    
    # Slider to pick a sample
    sample_idx = st.slider("Select Sample Index", 0, len(X_shap_sample)-1, 0, key="shap_slider")
    
    # Display the waterfall
    shap.plots.waterfall(shap_values[sample_idx], show=False)
    st.pyplot(plt.gcf())
    plt.clf()

with t_data:
    st.subheader("Dataset Analytics Dashboard")
    st.markdown("Explore trends and distributions across the high-entropy alloy dataset.")

    # Get data
    df = pipeline.X_final.copy()
    if pipeline.y is not None:
        df['YS (MPa)'] = pipeline.y.values
    else:
        st.error("Model target data (y) is missing!")
        st.stop()
    
    # Row 1: Key Relationships
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**1. Phase Stability: VEC vs Yield Strength**")
        fig_vec = px.scatter(
            df, 
            x="VEC_mean", 
            y="YS (MPa)", 
            color="VEC_mean",
            color_continuous_scale="Viridis",
            hover_data=df.columns,
            title="Impact of Valance Electron Concentration"
        )

        fig_vec.update_layout(height=400)
        st.plotly_chart(fig_vec, use_container_width=True)
        
    with col2:
        st.markdown("**2. Temperature vs Strength**")
        fig_temp = px.scatter(
            df, 
            x="Temperature", 
            y="YS (MPa)", 
            size="Grain_Size", # if available in dataset mapping
            color="YS (MPa)",
            title="Temperature & Grain Size Effects"
        )
        fig_temp.update_layout(height=400)
        st.plotly_chart(fig_temp, use_container_width=True)

    st.markdown("---")

    # Row 2: Correlation & Distribution
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("**3. Feature Correlations (Heatmap)**")
        corr = df.corr()
        fig_corr = px.imshow(
            corr, 
            text_auto=True, 
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with c2:
        st.markdown("**4. Yield Strength Distribution**")
        fig_dist = px.histogram(
            df, 
            x="YS (MPa)", 
            nbins=30, 
            title="Distribution of Yield Strengths in Dataset",
            color_discrete_sequence=['#4e73df']
        )
        st.plotly_chart(fig_dist, use_container_width=True)
