import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty

# ====================================================================
# DEFINE GLOBAL VARIABLES AND FILE PATHS
# ====================================================================

# Corrected file paths (assuming a 'Dataset' folder structure relative to model.py)
MPEA_FILE = "Dataset/MPEA_dataset.csv"
COMPOSITIONS_FILE = "Dataset/compositions.csv"

# Define standardized column names
FORMULA_COL = "FORMULA"
YS_COL = "PROPERTY: YS (MPa)"
TEMP_COL_OLD = "PROPERTY: Test temperature ($^\circ$C)" # Actual column name in file
TEMP_COL_NEW = "PROPERTY: Test temperature (C)" # Standardized name used in script
MODEL_SAVE_PATH = 'xgb_hea_model.pkl'

# ====================================================================
# MAIN EXECUTION BLOCK (Required for Matminer Multiprocessing)
# ====================================================================

if __name__ == '__main__':
    
    print("--- 1. Data Loading, Cleaning, and Merging ---")

    # --- Step 1.1: Load Data ---
    try:
        df_prop = pd.read_csv(MPEA_FILE)
        df_comp = pd.read_csv(COMPOSITIONS_FILE)
    except FileNotFoundError as e:
        print(f"Error: Required data file not found. Check path: {e}")
        raise

    # --- Step 1.2: Clean and Merge ---
    
    # Rename temperature column for easy access
    df_prop = df_prop.rename(columns={TEMP_COL_OLD: TEMP_COL_NEW})

    # Core Cleaning: Remove rows missing the target (YS) or critical input (Temp)
    df_cleaned = df_prop.dropna(subset=[YS_COL, TEMP_COL_NEW, FORMULA_COL])
    
    initial_rows = len(df_prop)
    print(f"Initial rows in MPEA dataset: {initial_rows}")
    print(f"Rows remaining after removing missing YS and Temp: {len(df_cleaned)}")

    # Merge cleaned properties with elemental compositions
    df_final = pd.merge(
        df_cleaned,
        df_comp.rename(columns={'Alloy name': FORMULA_COL}),
        on=FORMULA_COL,
        how='left'
    )
    print(f"Total entries after merging with compositions data: {len(df_final)}")
    
    # ====================================================================
    # 2. FEATURE ENGINEERING (Matminer)
    # ====================================================================

    print("\n--- 2. Feature Engineering (Matminer) ---")

    # --- Step 2.1: Convert Formula String to Composition Object ---
    # This is a prerequisite for Matminer's featurizers
    df_final = StrToComposition(target_col_id='Composition').featurize_dataframe(
        df_final, FORMULA_COL,
        ignore_errors=True
    )

    # --- Step 2.2: Define and Apply ElementProperty Featurizer ---
    ep_featurizer = ElementProperty(
        data_source='element', # Required argument
        features=['AtomicRadius', 'MendeleevNumber', 'MeltingT', 'Electronegativity', 'NValence'],
        stats=['mean', 'std_dev', 'max', 'min']
    )

    df_features = ep_featurizer.featurize_dataframe(
        df_final, 'Composition',
        n_jobs=1, # Fix for stable execution
        ignore_errors=True
    )

    # --- Step 2.3: Final Feature Matrix Preparation ---
    
    # Ensure target variable YS_MPa is numeric
    df_features[YS_COL] = pd.to_numeric(df_features[YS_COL], errors='coerce')
    
    # Drop rows where Y is still missing after final coercion
    df_model = df_features.dropna(subset=[YS_COL]) 

    # Identify all final numeric features (X columns)
    features_to_drop = [FORMULA_COL, 'Composition', YS_COL]
    X_cols = [col for col in df_model.columns if col not in features_to_drop and df_model[col].dtype in [np.float64, np.int64]]

    X = df_model[X_cols]
    Y = df_model[YS_COL]

    # Simple imputation for any remaining NaNs in feature columns
    X = X.fillna(X.mean())
    print(f"Total features generated and ready for model (X columns): {len(X.columns)}")

    # ====================================================================
    # 3. XGBOOST MODEL TRAINING AND EVALUATION
    # ====================================================================

    print("\n--- 3. XGBoost Model Training and Validation ---")
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # --- Step 3.1: Model Initialization and Training ---
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    xgb_model.fit(X_train, Y_train)
    print(f"Model Training Complete.")

    # --- Step 3.2: Evaluation ---
    Y_pred = xgb_model.predict(X_test)

    r2 = r2_score(Y_test, Y_pred)
    rmse = mean_squared_error(Y_test, Y_pred, squared=False)
    
    print(f"R-squared (R2) Score on Test Set: {r2:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} MPa")

    # --- Step 3.3: Save Model for Streamlit ---
    joblib.dump(xgb_model, MODEL_SAVE_PATH)
    print(f"\nSuccessfully saved the trained XGBoost model to: {MODEL_SAVE_PATH}")
    
    # --- Step 3.4: Feature Importance (Optional Final Output) ---
    importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
    print("\nTop 5 Most Important Features:")
    print(importance.nlargest(5).to_markdown())
