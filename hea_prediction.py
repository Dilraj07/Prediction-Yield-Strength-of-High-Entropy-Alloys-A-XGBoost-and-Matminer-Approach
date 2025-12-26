import pandas as pd
import numpy as np
import re
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns


# --- 1. Elemental Property Database ---
# Dictionary of key physical properties for common HEA elements
element_data = {
    'Al': {'r': 1.43, 'vec': 3, 'x': 1.61, 'tm': 933},
    'Co': {'r': 1.25, 'vec': 9, 'x': 1.88, 'tm': 1768},
    'Cr': {'r': 1.28, 'vec': 6, 'x': 1.66, 'tm': 2180},
    'Cu': {'r': 1.28, 'vec': 11, 'x': 1.90, 'tm': 1357},
    'Fe': {'r': 1.26, 'vec': 8, 'x': 1.83, 'tm': 1811},
    'Mn': {'r': 1.27, 'vec': 7, 'x': 1.55, 'tm': 1519},
    'Ni': {'r': 1.24, 'vec': 10, 'x': 1.91, 'tm': 1728},
    'Ti': {'r': 1.47, 'vec': 4, 'x': 1.54, 'tm': 1941},
    'V':  {'r': 1.34, 'vec': 5, 'x': 1.63, 'tm': 2183},
    'Zn': {'r': 1.34, 'vec': 12, 'x': 1.65, 'tm': 692},
    'Zr': {'r': 1.60, 'vec': 4, 'x': 1.33, 'tm': 2128},
    'Hf': {'r': 1.59, 'vec': 4, 'x': 1.30, 'tm': 2506},
    'Mo': {'r': 1.39, 'vec': 6, 'x': 2.16, 'tm': 2896},
    'Nb': {'r': 1.46, 'vec': 5, 'x': 1.60, 'tm': 2750},
    'Ta': {'r': 1.46, 'vec': 5, 'x': 1.50, 'tm': 3290},
    'W':  {'r': 1.39, 'vec': 6, 'x': 2.36, 'tm': 3695},
    'C':  {'r': 0.70, 'vec': 4, 'x': 2.55, 'tm': 3800},
    'B':  {'r': 0.85, 'vec': 3, 'x': 2.04, 'tm': 2349},
    'Si': {'r': 1.10, 'vec': 4, 'x': 1.90, 'tm': 1687},
    # Add more elements as needed
}

# --- 2. Helper Functions ---

def parse_formula(formula):
    """Parses chemical formula (e.g., 'Al1Co1Cr1') into atomic fractions."""
    pattern = r"([A-Z][a-z]*)([\d\.]*)?"
    matches = re.findall(pattern, str(formula))
    composition = {}
    for elt, amt in matches:
        if elt not in element_data: continue
        try:
            amount = float(amt) if amt else 1.0
        except:
            amount = 1.0
        composition[elt] = composition.get(elt, 0) + amount
    
    total = sum(composition.values())
    if total == 0: return None
    for k in composition: composition[k] /= total
    return composition

def calculate_features(composition, temperature):
    """Calculates physics-based features (VEC, Delta, Mixing Entropy, etc.)."""
    if composition is None: return None
    
    vec_avg, r_avg, x_avg, tm_avg, s_mix = 0, 0, 0, 0, 0
    
    # Calculate weighted averages
    for elt, frac in composition.items():
        p = element_data[elt]
        vec_avg += frac * p['vec']
        r_avg += frac * p['r']
        x_avg += frac * p['x']
        tm_avg += frac * p['tm']
        if frac > 0: 
            s_mix += -8.314 * frac * np.log(frac) # Gas constant R = 8.314
            
    # Calculate differences (Delta parameters)
    r_delta_sq = sum(frac * (1 - element_data[elt]['r']/r_avg)**2 for elt, frac in composition.items())
    x_delta_sq = sum(frac * (element_data[elt]['x'] - x_avg)**2 for elt, frac in composition.items())
    
    delta_r = np.sqrt(r_delta_sq) * 100
    delta_x = np.sqrt(x_delta_sq)
    
    return [vec_avg, r_avg, delta_r, x_avg, delta_x, tm_avg, s_mix, temperature]

feature_names = ['VEC_mean', 'R_mean', 'Delta_R', 'X_mean', 'X_diff', 'Tm_mean', 'S_mix', 'Temperature']

# --- 3. Data Processing Pipeline ---

# Load Dataset
df = pd.read_csv("MPEA_dataset.csv")

# Filter rows with valid Target
df = df.dropna(subset=['PROPERTY: YS (MPa)'])

X_data, valid_indices = [], []

print("Generating features...")
for idx, row in df.iterrows():
    # Handle missing temperature (default to room temp 25C if missing)
    temp = row['PROPERTY: Test temperature ($^\circ$C)']
    temp = 25.0 if pd.isna(temp) else temp
    
    # Parse Formula and Calc Features
    feats = calculate_features(parse_formula(row['FORMULA']), temp)
    if feats:
        X_data.append(feats)
        valid_indices.append(idx)

# Create Feature DataFrame
X = pd.DataFrame(X_data, columns=feature_names)
df_valid = df.loc[valid_indices].reset_index(drop=True)
y = df_valid['PROPERTY: YS (MPa)']

# Handle Grain Size (Impute missing with median)
imp_grain = SimpleImputer(strategy='median')
gs = pd.to_numeric(df_valid['PROPERTY: grain size (mm)'], errors='coerce')
X['Grain_Size'] = imp_grain.fit_transform(gs.values.reshape(-1, 1))

# Handle Processing Method (One-Hot Encoding)
proc = df_valid['PROPERTY: Processing method'].fillna('OTHER').astype(str).str.upper()
# Simplify categories
proc = proc.apply(lambda x: 'CAST' if 'CAST' in x else ('ANNEAL' if 'ANNEAL' in x else 'OTHER'))
enc_proc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
proc_enc = enc_proc.fit_transform(proc.values.reshape(-1, 1))
proc_cols = enc_proc.get_feature_names_out(['Proc'])
X_proc = pd.DataFrame(proc_enc, columns=proc_cols)

# Final Feature Matrix
X_final = pd.concat([X, X_proc], axis=1)

# --- 4. Model Training ---

print("Training Model...")
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.15, random_state=42)

# Gradient Boosting (XGBoost equivalent in Scikit-Learn)
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Evaluation
score = model.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
print(f"Model Trained! Test R2 Score: {score:.3f}")
print(f"Test RMSE: {rmse:.2f} MPa")

# --- 5. Interactive Prediction Function ---

def predict_yield_strength():
    print("\n--- Predict Yield Strength for New Alloy ---")
    formula = input("Enter Alloy Formula (e.g., Co1Cr1Fe1Ni1): ")
    temp = float(input("Enter Test Temperature (°C): "))
    grain = float(input("Enter Grain Size (um) [approx 50 if unknown]: "))
    method = input("Enter Processing Method (CAST / ANNEAL / OTHER): ").upper()
    
    # 1. Calc Formula Features
    comp = parse_formula(formula)
    if not comp:
        print("Error: Could not parse elements.")
        return
    base_feats = calculate_features(comp, temp)
    
    # 2. Build DataFrame row
    input_df = pd.DataFrame([base_feats], columns=feature_names)
    input_df['Grain_Size'] = grain
    
    # 3. Encode Processing Method
    method_clean = 'CAST' if 'CAST' in method else ('ANNEAL' if 'ANNEAL' in method else 'OTHER')
    proc_vec = enc_proc.transform([[method_clean]])
    proc_df = pd.DataFrame(proc_vec, columns=proc_cols)
    
    # 4. Combine and Predict
    final_vec = pd.concat([input_df, proc_df], axis=1)
    pred_ys = model.predict(final_vec)[0]
    
    print(f"\n>>> Predicted Yield Strength: {pred_ys:.2f} MPa")
    print(f">>> Key Indicators: VEC={base_feats[0]:.2f}, Delta_R={base_feats[2]:.2f}%")

# Run this line to start prediction:
predict_yield_strength()

# --- IMPROVED FEATURE IMPORTANCE CODE ---

# 1. Get the feature names directly from the trained model
# This handles the extra columns created by One-Hot Encoding automatically
try:
    # Try getting names if using Scikit-Learn wrapper (XGBRegressor)
    feature_names = model.feature_names_in_
except:
    # Fallback for raw XGBoost or if names are missing
    feature_names = [f"Feature {i}" for i in range(len(model.feature_importances_))]

# 2. Get the importance values
importance = model.feature_importances_

# 3. Create the DataFrame (Now lengths are guaranteed to match)
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

# 4. Sort and Plot
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, hue='Feature', palette='viridis', legend=False)
plt.title('Feature Importance: What Drives Yield Strength?', fontsize=16)
plt.xlabel('Relative Importance', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

print("\n>>> Top Drivers of Strength:")
print(feature_importance_df.head(3))