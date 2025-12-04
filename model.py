import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty

# ====================================================================
# PROJECT SETUP AND DATA CLEANING (Phase 1)
# ====================================================================

# Define file names and critical columns as provided by the user
MPEA_FILE = "Dataset/MPEA_dataset.csv"
COMPOSITIONS_FILE = "Dataset/compositions.csv"
FORMULA_COL = "FORMULA"
YS_COL = "PROPERTY: YS (MPa)"
TEMP_COL = "PROPERTY: Test temperature (C)" # Using a simplified version of the column name for clarity


if __name__ == '__main__':
    print("--- 1. Data Loading and Initial Cleaning ---")

    # --- Step 1.1: Load Data ---
    try:
        df_prop = pd.read_csv(MPEA_FILE)
        df_comp = pd.read_csv(COMPOSITIONS_FILE)
    except FileNotFoundError as e:
        print(f"Error: Required file not found. Ensure {MPEA_FILE} and {COMPOSITIONS_FILE} are in the current directory.")
        raise e

    # --- Step 1.2: Perform Initial Cleaning on Properties Data ---
    # Standardize column names based on previous inspection if necessary (handling special chars)
    # Assuming the column names for YS and Temperature are close enough for direct access or minor cleanup:

    # Renaming the Temperature column based on previous successful execution context for robustness
    df_prop = df_prop.rename(columns={r'PROPERTY: Test temperature ($^\circ$C)': TEMP_COL})
    initial_rows = len(df_prop)

    # Remove any alloy with missing properties (Yield Strength and Test Temperature)
    df_cleaned = df_prop.dropna(subset=[YS_COL, TEMP_COL, FORMULA_COL])

    print(f"Initial rows in MPEA dataset: {initial_rows}")
    print(f"Rows remaining after removing missing YS and Temp: {len(df_cleaned)}")

    # --- Step 1.3: Merge DataFrames ---
    # Merge the cleaned experimental data with the full elemental breakdown data
    # The key for merging is the composition formula/alloy name
    df_final = pd.merge(
        df_cleaned,
        df_comp.rename(columns={'Alloy name': FORMULA_COL}),
        on=FORMULA_COL,
        how='left'
    )

    print(f"Total entries after merging with compositions data: {len(df_final)}")
    print(f"The combined, cleaned data is ready for feature engineering.")


    # ====================================================================
    # FEATURE ENGINEERING (Matminer - Phase 2)
    # ====================================================================

    print("\n--- 2. Feature Engineering (Matminer) ---")

    # --- Step 2.1: Convert Formula String to Composition Object (Matminer Requirement) ---
    # Matminer needs a composition object to calculate features
    df_final = StrToComposition(target_col_id='Composition').featurize_dataframe(
        df_final, FORMULA_COL
    )

    # --- Step 2.2: Define and Apply Matminer Featurizer ---
    # We use ElementProperty to calculate mean, variance, max, min, etc., for key atomic properties
    # These features directly address the scientific drivers of yield strength (entropy, size, VEC)

    ep_featurizer = ElementProperty(
    # --- FIX: ADD REQUIRED 'data_source' ARGUMENT ---
    data_source='element', # 'element' is a good general purpose source
    
    features=['AtomicRadius', 'MendeleevNumber', 'MeltingT', 'Electronegativity', 'NValence'],
    stats=['mean', 'std_dev', 'max', 'min']
    )

    df_features = ep_featurizer.featurize_dataframe(
        df_final, 'Composition'
    )

    # --- Step 2.3: Add Key Features Manually (e.g., Mixing Entropy) ---
    # Although some are calculated by Matminer, calculating them directly ensures standard names/formats
    # For demonstration, we'll assume a robust featurizer from a custom Matminer list is used.
    # The previous ElementProperty features will serve as our final feature matrix.

    # Drop auxiliary columns created during featurization
    features_to_drop = [FORMULA_COL, 'Composition']
    df_features = df_features.drop(columns=features_to_drop, errors='ignore')

    # Identify all numeric columns for the final feature matrix X
    X_cols = [col for col in df_features.columns if df_features[col].dtype in [np.float64, np.int64] and col != YS_COL]

    print(f"Total features generated for the model (X columns): {len(X_cols)}")


    # ====================================================================
    # MODEL TRAINING AND EVALUATION (XGBoost - Phase 3)
    # ====================================================================

    print("\n--- 3. XGBoost Model Training and Validation ---")

    # --- Step 3.1: Final Data Cleaning & Split ---
    # Ensure target variable YS_MPa is numeric
    df_features[YS_COL] = pd.to_numeric(df_features[YS_COL], errors='coerce')

    # Drop any rows where target (Y) is missing after final cleaning
    df_model = df_features.dropna(subset=[YS_COL])

    X = df_model[X_cols]
    Y = df_model[YS_COL]

    # Fill remaining NaNs in features with the mean (simple imputation)
    X = X.fillna(X.mean())

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # --- Step 3.2: Model Initialization and Training ---
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    xgb_model.fit(X_train, Y_train)

    # --- Step 3.3: Prediction and Evaluation ---
    Y_pred = xgb_model.predict(X_test)

    r2 = r2_score(Y_test, Y_pred)
    rmse = mean_squared_error(Y_test, Y_pred, squared=False)
    mae = mean_absolute_error(Y_test, Y_pred)

    print(f"Model Training Complete.")
    print(f"R-squared (R2) Score on Test Set: {r2:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} MPa")
    print(f"Mean Absolute Error (MAE): {mae:.2f} MPa")

    # --- Step 3.4: Feature Importance (Scientific Interpretability) ---
    importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
    print("\nTop 5 Most Important Features:")
    print(importance.nlargest(5).to_markdown())

print("\n--- Project Workflow Complete ---")