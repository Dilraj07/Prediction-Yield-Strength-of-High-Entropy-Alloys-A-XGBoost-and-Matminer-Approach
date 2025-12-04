
import os
# --- Set environment variables BEFORE importing heavy numerical libs ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

# Matminer imports (delayed to allow env vars set first)
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty

# ====================================================================
# DEFINE GLOBAL VARIABLES AND FILE PATHS
# ====================================================================
MPEA_FILE = "Dataset/MPEA_dataset.csv"
COMPOSITIONS_FILE = "Dataset/compositions.csv"

FORMULA_COL = "FORMULA"
YS_COL = "PROPERTY: YS (MPa)"
TEMP_COL_OLD = "PROPERTY: Test temperature ($^\\circ$C)"
TEMP_COL_NEW = "PROPERTY: Test temperature (C)"
MODEL_SAVE_PATH = 'xgb_hea_model.pkl'

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================
def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def try_get_column(df, candidates, rename_to=None):
    """Return the first matching column name from candidates; optionally rename in-place."""
    for c in candidates:
        if c in df.columns:
            if rename_to:
                df.rename(columns={c: rename_to}, inplace=True)
            return rename_to or c
    return None

# ====================================================================
# MAIN EXECUTION BLOCK (Required for Matminer Multiprocessing)
# ====================================================================
if __name__ == '__main__':
    print("--- 1. Data Loading, Cleaning, and Merging ---")

    # --- Step 1.1: Load Data ---
    try:
        df_prop = safe_read_csv(MPEA_FILE)
        df_comp = safe_read_csv(COMPOSITIONS_FILE)
    except Exception as e:
        print(f"ERROR: could not load datasets: {e}")
        raise

    # Defensive: show columns to help debug
    print("MPEA columns:", list(df_prop.columns)[:20])
    print("Compositions columns:", list(df_comp.columns)[:20])

    # --- Step 1.2: Column handling & Cleaning ---
    # Normalize formula column: try common candidates in compositions file
    # If compositions uses 'Alloy name' rename to FORMULA_COL for merging
    if 'Alloy name' in df_comp.columns and FORMULA_COL not in df_comp.columns:
        df_comp = df_comp.rename(columns={'Alloy name': FORMULA_COL})

    # Ensure temperature column available: try both old and new names
    temp_col_found = try_get_column(df_prop, [TEMP_COL_OLD, TEMP_COL_NEW], rename_to=TEMP_COL_NEW)
    if temp_col_found is None:
        print(f"Warning: temperature column not found (tried '{TEMP_COL_OLD}' and '{TEMP_COL_NEW}'). Continuing but check your data.")
    else:
        print(f"Using temperature column: {TEMP_COL_NEW}")

    # Ensure formula column exists in df_prop (some datasets name it differently)
    if FORMULA_COL not in df_prop.columns:
        # try common alternatives
        alt_candidates = ['Alloy', 'alloy', 'Formula', 'formula', 'composition', 'Composition']
        found = try_get_column(df_prop, alt_candidates, rename_to=FORMULA_COL)
        if found is None:
            raise KeyError(f"Could not find formula column in property dataset. Expected '{FORMULA_COL}' or alternatives: {alt_candidates}")

    # Ensure YS column exists
    if YS_COL not in df_prop.columns:
        alt_y_candidates = ['YS_MPa', 'YS (MPa)', 'Yield Strength (MPa)', 'yield_strength']
        found_y = try_get_column(df_prop, alt_y_candidates, rename_to=YS_COL)
        if found_y is None:
            raise KeyError(f"Could not find target column '{YS_COL}' or alternatives: {alt_y_candidates}")

    # Drop entries missing required core columns
    df_cleaned = df_prop.dropna(subset=[YS_COL, FORMULA_COL])
    print(f"Initial rows in MPEA dataset: {len(df_prop)}")
    print(f"Rows remaining after removing missing YS and Formula: {len(df_cleaned)}")

    # Merge with compositions (left join)
    df_final = pd.merge(
        df_cleaned,
        df_comp[[FORMULA_COL] + [c for c in df_comp.columns if c != FORMULA_COL]],
        on=FORMULA_COL,
        how='left'
    )
    print(f"Total entries after merging with compositions data: {len(df_final)}")

    # ====================================================================
    # 2. FEATURE ENGINEERING (Matminer) - single-threaded safe calls
    # ====================================================================
    print("\n--- 2. Feature Engineering (Matminer) ---")
    # Convert formula to Composition object (n_jobs=1 to avoid multiprocessing)
    try:
        df_final = StrToComposition(
            target_col_id='Composition'
        ).featurize_dataframe(df_final, FORMULA_COL, n_jobs=1, ignore_errors=True)
    except TypeError:
        # Some matminer versions expect different signature - try without n_jobs
        df_final = StrToComposition(target_col_id='Composition').featurize_dataframe(
            df_final, FORMULA_COL, ignore_errors=True
        )
    except Exception as e:
        print("ERROR in StrToComposition.featurize_dataframe:", e)
        raise

    # Create ElementProperty featurizer. Use from_preset if available, fallback to standard init.
    try:
        # Preferred: use a preset (magpie) which is stable across matminer versions
        ep_featurizer = ElementProperty.from_preset("magpie")
    except Exception:
        # Fallback: explicit construction
        ep_featurizer = ElementProperty(
            data_source='magpie',
            features=['AtomicRadius', 'MendeleevNumber', 'MeltingT', 'Electronegativity', 'NValence'],
            stats=['mean', 'std_dev', 'max', 'min']
        )

    # Featurize (n_jobs=1)
    try:
        df_features = ep_featurizer.featurize_dataframe(df_final, 'Composition', n_jobs=1, ignore_errors=True)
    except TypeError:
        df_features = ep_featurizer.featurize_dataframe(df_final, 'Composition', ignore_errors=True)
    except Exception as e:
        print("ERROR in ElementProperty.featurize_dataframe:", e)
        raise

    # ====================================================================
    # 2.5 Post-featurization cleaning
    # ====================================================================
    # Ensure target numeric
    df_features[YS_COL] = pd.to_numeric(df_features[YS_COL], errors='coerce')
    df_model = df_features.dropna(subset=[YS_COL]).reset_index(drop=True)

    print(f"Rows available for modelling after dropping missing YS: {len(df_model)}")

    # Identify numeric feature columns; exclude identifiers
    features_to_drop = [FORMULA_COL, 'Composition', YS_COL]
    X_cols = [col for col in df_model.columns
              if col not in features_to_drop and
                 (np.issubdtype(df_model[col].dtype, np.number))]

    X = df_model[X_cols].copy()
    Y = df_model[YS_COL].copy()

    # If no features found, stop with informative message
    if X.shape[1] == 0:
        raise RuntimeError("No numeric features found after featurization. Check featurizer output and column types.")

    # Impute simple NaNs in feature columns with column mean
    X = X.fillna(X.mean())

    print(f"Total features generated and ready for model (X columns): {len(X.columns)}")

    # If dataset is very large, warn user
    if len(X) > 20000:
        print("WARNING: dataset is large (>20k rows). This may consume a lot of memory and take long to featurize/train.")

    # ====================================================================
    # 3. XGBOOST MODEL TRAINING AND EVALUATION
    # ====================================================================
    print("\n--- 3. XGBoost Model Training and Validation ---")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Create model - force CPU usage and single-threaded training to avoid OMP conflicts
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        n_jobs=1,
        tree_method="hist",
        random_state=42
    )

    # Fit
    try:
        xgb_model.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], verbose=False)
    except Exception as e:
        print("ERROR during XGBoost training:", e)
        raise

    print("Model Training Complete.")

    # Predictions and evaluation
    Y_pred = xgb_model.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    rmse = mean_squared_error(Y_test, Y_pred, squared=False)
    mae = np.mean(np.abs(Y_test - Y_pred))

    print(f"R-squared (R2) Score on Test Set: {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} MPa")
    print(f"Mean Absolute Error (MAE): {mae:.4f} MPa")

    # Save model
    joblib.dump(xgb_model, MODEL_SAVE_PATH)
    print(f"\nSuccessfully saved the trained XGBoost model to: {MODEL_SAVE_PATH}")

    # Feature importance
    try:
        importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
        print("\nTop 10 Most Important Features:")
        print(importance.sort_values(ascending=False).head(10).to_string())
    except Exception as e:
        print("Could not compute feature importance:", e)
