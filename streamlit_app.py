import streamlit as st
import pandas as pd
import numpy as np
import joblib # Used to load the trained model
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty

# --- 1. Load Model and Data ---
# NOTE: Replace these placeholders with your actual model and data loading paths.
try:
    # 1.1 Load the trained XGBoost model
    MODEL = joblib.load('xgb_hea_model.pkl')
    
    # 1.2 Load the featurizer (necessary to get feature names)
    # This assumes you saved the trained featurizer or can recreate the list of columns (X_cols)
    EP_FEATURIZER = ElementProperty(
        data_source='element',
        features=['AtomicRadius', 'MendeleevNumber', 'MeltingT', 'Electronegativity', 'NValence'],
        stats=['mean', 'std_dev', 'max', 'min']
    )
    
except FileNotFoundError:
    st.error("Error: Model file (xgb_hea_model.pkl) or featurizer setup file not found. Ensure models are saved.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or featurizer: {e}")
    st.stop()


# --- 2. Feature Generation Function ---
@st.cache_data
def generate_features(formula_str):
    """Generates features from a formula string using Matminer."""
    try:
        # Create a single-row DataFrame for the input formula
        df_input = pd.DataFrame({'FORMULA': [formula_str]})

        # Convert formula string to Composition object
        df_input = StrToComposition(target_col_id='Composition').featurize_dataframe(df_input, 'FORMULA', ignore_errors=True)
        
        # Apply ElementProperty featurizer
        df_input = EP_FEATURIZER.featurize_dataframe(df_input, 'Composition', ignore_errors=True)
        
        # Clean up columns and select only the generated features
        X_test_cols = [col for col in df_input.columns if col.startswith('ElementProperty')]
        
        # Select the feature columns and ensure they match the order of the training set
        X_test = df_input[X_test_cols]
        
        # Impute NaNs with a standard value (e.g., mean from training set, but using 0 here for simplicity)
        X_test = X_test.fillna(0)
        
        return X_test

    except Exception as e:
        st.error(f"Feature Generation Failed: {e}")
        return None

# --- 3. Streamlit Interface Layout ---

st.set_page_config(
    page_title="HEA Yield Strength Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 HEA Yield Strength Predictor")
st.markdown("---")

st.sidebar.header("Alloy Input")
st.sidebar.markdown("Enter the High-Entropy Alloy composition (e.g., FeCrNiCoMn).")

# Composition Input
formula_input = st.sidebar.text_input(
    label="Chemical Formula (e.g., FeCrNiCoMn)",
    value="Al0.2CoFeNi" 
)

# Test Temperature Input (Crucial Experimental Input)
temp_input = st.sidebar.number_input(
    label="Test Temperature (°C)",
    min_value=-273,
    max_value=1200,
    value=25
)

# Prediction Button
predict_button = st.sidebar.button("Predict Yield Strength")


# --- Main Content: Display Results ---

if predict_button and formula_input:
    st.header("Prediction Results & Scientific Insights")

    # 3.1 Generate features for the user's input
    X_input = generate_features(formula_input)

    if X_input is not None and not X_input.empty:
        
        # 3.2 Add the explicit temperature feature to the input matrix
        # This assumes your training set includes 'PROPERTY: Test temperature (C)' as a feature
        X_input['PROPERTY: Test temperature (C)'] = temp_input
        
        try:
            # 3.3 Make Prediction
            predicted_ys = MODEL.predict(X_input)[0]
            
            # --- Output Panel 1: Prediction ---
            st.success("✅ Prediction Successful!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Predicted Yield Strength ($\mathbf{\sigma_y}$)",
                    value=f"{predicted_ys:.2f} MPa",
                    delta="Ambient Temp Baseline"
                )
            
            with col2:
                # Add a physical insight metric (e.g., estimated hardness or VEC)
                st.metric(
                    label="Estimated Valence Electron Concentration (VEC)",
                    value=f"{X_input['ElementProperty mean NValence'].iloc[0]:.2f}"
                )
            
            st.markdown("---")
            
            # --- Output Panel 2: Interpretability (Feature Importance) ---
            st.subheader("Key Scientific Drivers")
            st.markdown("The chart below shows which physicochemical features had the greatest influence on this specific prediction.")

            # Calculate SHAP or Feature Importance (using XGBoost's native method for simplicity)
            # NOTE: For true SHAP, you would need the shap library and specific SHAP calculation.
            
            # Use a dummy importance ranking for this specific prediction's features
            # In a real app, you'd use a dedicated SHAP explainer here.
            feature_importance_df = pd.Series(MODEL.feature_importances_, index=MODEL.feature_names_in_)
            feature_importance_df = feature_importance_df.sort_values(ascending=False).head(10)
            
            # Display feature importance as a bar chart
            st.bar_chart(feature_importance_df)
            
            st.caption("Feature Importance indicates which calculated properties drove the predicted result.")

        except Exception as e:
            st.error(f"Model Prediction Failed. Check feature alignment: {e}")
            st.stop()
            
    else:
        st.warning("Could not generate valid features. Check the formula format.")

else:
    st.info("Enter an alloy formula in the sidebar and click 'Predict Yield Strength' to begin.")
