import streamlit as st
import joblib
import pandas as pd

from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty

# ---------------------------
# Load models and featurizers
# ---------------------------

@st.cache_resource
def load_models():
    fcc_model = joblib.load("FCC_YS_xgb.pkl")
    bcc_model = joblib.load("BCC_YS_xgb.pkl")

    fcc_featurizer = joblib.load("fcc_featurizer.pkl")
    bcc_featurizer = joblib.load("bcc_featurizer.pkl")

    fcc_features = joblib.load("fcc_feature_names.pkl")
    bcc_features = joblib.load("bcc_feature_names.pkl")

    return fcc_model, bcc_model, fcc_featurizer, bcc_featurizer, fcc_features, bcc_features


(
    fcc_model,
    bcc_model,
    fcc_featurizer,
    bcc_featurizer,
    fcc_features,
    bcc_features,
) = load_models()


# ---------------------------
# Featurization function
# ---------------------------

def featurize_input(formula, phase):
    df = pd.DataFrame({"FORMULA": [formula]})

    # Convert string → Composition object
    stc = StrToComposition(target_col_id="composition_obj")
    df = stc.featurize_dataframe(df, "FORMULA", ignore_errors=True)

    if phase == "FCC":
        df = fcc_featurizer.featurize_dataframe(df, "composition_obj", ignore_errors=True)
        df = df[fcc_features]

    elif phase == "BCC":
        df = bcc_featurizer.featurize_dataframe(df, "composition_obj", ignore_errors=True)
        df = df[bcc_features]

    return df


# ---------------------------
# Streamlit UI
# ---------------------------

st.title("HEA Yield Strength Predictor")

st.write("Machine learning model using XGBoost + Matminer")
st.write("Currently supports BCC and FCC HEAs at room temperature")

formula = st.text_input(
    "Enter HEA Composition (example: Co1Cr1Fe1Mn1Ni1 or Al0.3CoCrFeNi)"
)

phase = st.selectbox(
    "Crystal Structure",
    ["FCC", "BCC"]
)

predict_button = st.button("Predict Yield Strength (MPa)")

# ---------------------------
# Prediction
# ---------------------------

if predict_button:

    if formula.strip() == "":
        st.error("Please enter a valid composition")
    else:
        try:
            X = featurize_input(formula, phase)

            if phase == "FCC":
                model = fcc_model
            else:
                model = bcc_model

            ys_pred = model.predict(X)[0]

            st.subheader("Predicted Yield Strength")
            st.success(f"≈ {ys_pred:.2f} MPa")

            if phase == "FCC":
                st.info("Note: FCC model error is higher due to dataset scatter.")

        except Exception as e:
            st.error("Failed to featurize or predict. Check composition formatting.")
            st.code(str(e))
