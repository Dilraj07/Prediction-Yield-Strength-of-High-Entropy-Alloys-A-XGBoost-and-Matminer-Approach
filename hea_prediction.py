import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr

# --- CONFIGURATION FOR PUBLICATION FIGURES ---
# Sets font to generic sans-serif (Arial-like) which is standard for journals
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 12

# --- 1. ELEMENTAL DATABASE (UNCHANGED) ---
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
}

# --- 2. CORE HELPER FUNCTIONS ---
def parse_formula(formula):
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
    if composition is None: return None
    vec_avg, r_avg, x_avg, tm_avg, s_mix = 0, 0, 0, 0, 0
    for elt, frac in composition.items():
        p = element_data[elt]
        vec_avg += frac * p['vec']
        r_avg += frac * p['r']
        x_avg += frac * p['x']
        tm_avg += frac * p['tm']
        if frac > 0: 
            s_mix += -8.314 * frac * np.log(frac)
            
    r_delta_sq = sum(frac * (1 - element_data[elt]['r']/r_avg)**2 for elt, frac in composition.items())
    x_delta_sq = sum(frac * (element_data[elt]['x'] - x_avg)**2 for elt, frac in composition.items())
    
    delta_r = np.sqrt(r_delta_sq) * 100
    delta_x = np.sqrt(x_delta_sq)
    return [vec_avg, r_avg, delta_r, x_avg, delta_x, tm_avg, s_mix, temperature]

feature_names = ['VEC_mean', 'R_mean', 'Delta_R', 'X_mean', 'X_diff', 'Tm_mean', 'S_mix', 'Temperature']

# --- 3. DATA PROCESSING & VALIDATION CLASS ---
class HEAModelPipeline:
    def __init__(self, csv_path):
        self.df_raw = pd.read_csv(csv_path)
        self.model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
        self.enc_proc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.X_final = None
        self.y = None
        self.feature_cols = []
        
    def preprocess(self):
        print(">>> Processing Data & Engineering Features...")
        df = self.df_raw.dropna(subset=['PROPERTY: YS (MPa)'])
        X_data, valid_indices = [], []

        for idx, row in df.iterrows():
            temp = row['PROPERTY: Test temperature ($^\circ$C)']
            temp = 25.0 if pd.isna(temp) else temp
            feats = calculate_features(parse_formula(row['FORMULA']), temp)
            if feats:
                X_data.append(feats)
                valid_indices.append(idx)

        X = pd.DataFrame(X_data, columns=feature_names)
        df_valid = df.loc[valid_indices].reset_index(drop=True)
        self.y = df_valid['PROPERTY: YS (MPa)']

        # Impute Grain Size
        imp_grain = SimpleImputer(strategy='median')
        gs = pd.to_numeric(df_valid['PROPERTY: grain size (mm)'], errors='coerce')
        X['Grain_Size'] = imp_grain.fit_transform(gs.values.reshape(-1, 1))

        # One-Hot Encoding
        proc = df_valid['PROPERTY: Processing method'].fillna('OTHER').astype(str).str.upper()
        proc = proc.apply(lambda x: 'CAST' if 'CAST' in x else ('ANNEAL' if 'ANNEAL' in x else 'OTHER'))
        proc_enc = self.enc_proc.fit_transform(proc.values.reshape(-1, 1))
        proc_cols = self.enc_proc.get_feature_names_out(['Proc'])
        X_proc = pd.DataFrame(proc_enc, columns=proc_cols)

        self.X_final = pd.concat([X, X_proc], axis=1)
        self.feature_cols = self.X_final.columns.tolist()

    def train_and_validate(self):
        print("\n>>> Starting Rigorous Validation...")
        X_train, X_test, y_train, y_test = train_test_split(self.X_final, self.y, test_size=0.15, random_state=42)
        
        # 1. Train
        self.model.fit(X_train, y_train)
        y_pred_test = self.model.predict(X_test)
        y_pred_train = self.model.predict(X_train)
        
        # 2. Cross-Validation (Standard for Papers)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, self.X_final, self.y, cv=kf, scoring='neg_root_mean_squared_error')
        cv_rmse = -np.mean(cv_scores)
        
        print(f"   Train R2: {r2_score(y_train, y_pred_train):.3f}")
        print(f"   Test R2:  {r2_score(y_test, y_pred_test):.3f}")
        print(f"   CV RMSE:  {cv_rmse:.2f} MPa (5-Fold Average)")

        return X_test, y_test, y_pred_test

    def generate_paper_figures(self, X_test, y_test, y_pred):
        print("\n>>> Generating Publication-Quality Figures...")
        
        # FIGURE 1: Predicted vs Actual (Parity Plot)
        plt.figure(figsize=(7, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, color='#2b7bba', edgecolors='w', s=70, label='Test Data')
        
        # Perfect prediction line
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
        
        # Annotate Metrics directly on plot
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        plt.text(0.05, 0.9, f'$R^2 = {r2:.2f}$\n$RMSE = {rmse:.0f} MPa$', 
                 transform=plt.gca().transAxes, fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        plt.xlabel('Experimental Yield Strength (MPa)', fontsize=12)
        plt.ylabel('Predicted Yield Strength (MPa)', fontsize=12)
        plt.title('Parity Plot: Model Prediction Performance', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('Fig1_Parity_Plot.png', dpi=300)
        print("   [Saved] Fig1_Parity_Plot.png")

        # FIGURE 2: Feature Correlation Heatmap (EDA)
        plt.figure(figsize=(10, 8))
        # Select only numeric physical features for correlation
        phys_feats = self.X_final[feature_names + ['Grain_Size']]
        corr_matrix = phys_feats.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Hide upper triangle
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                    square=True, linewidths=.5, cbar_kws={"shrink": .8})
        plt.title('Pearson Correlation of Input Features', fontsize=14)
        plt.tight_layout()
        plt.savefig('Fig2_Correlation_Matrix.png', dpi=300)
        print("   [Saved] Fig2_Correlation_Matrix.png")

        # FIGURE 3: Feature Importance (From your previous code)
        importance = self.model.feature_importances_
        fi_df = pd.DataFrame({'Feature': self.feature_cols, 'Importance': importance})
        fi_df = fi_df.sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Importance', y='Feature', data=fi_df, hue='Feature', palette='viridis', legend=False)
        plt.title('Feature Importance Analysis (Gradient Boosting)', fontsize=14)
        plt.xlabel('Relative Importance (Gini Impurity)', fontsize=12)
        plt.tight_layout()
        plt.savefig('Fig3_Feature_Importance.png', dpi=300)
        print("   [Saved] Fig3_Feature_Importance.png")

    def manual_test(self):
        print("\n--- Manual Test Mode ---")
        formula = input("Enter Alloy Formula (e.g., Al1Co1Cr1Fe1Ni1): ")
        temp = float(input("Enter Temp (°C): "))
        grain = float(input("Enter Grain Size (um): "))
        method = input("Method (CAST/ANNEAL): ").upper()
        
        # Calculation logic same as before...
        comp = parse_formula(formula)
        base = calculate_features(comp, temp)
        row = pd.DataFrame([base], columns=feature_names)
        row['Grain_Size'] = grain
        
        method_clean = 'CAST' if 'CAST' in method else ('ANNEAL' if 'ANNEAL' in method else 'OTHER')
        proc_vec = self.enc_proc.transform([[method_clean]])
        proc_df = pd.DataFrame(proc_vec, columns=self.enc_proc.get_feature_names_out(['Proc']))
        
        final = pd.concat([row, proc_df], axis=1)
        pred = self.model.predict(final)[0]
        print(f"\n>>> Prediction: {pred:.2f} MPa")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Initialize and run
    pipeline = HEAModelPipeline("MPEA_dataset.csv") # <--- Ensure filename matches!
    pipeline.preprocess()
    X_test, y_test, y_pred = pipeline.train_and_validate()
    pipeline.generate_paper_figures(X_test, y_test, y_pred)
    
    # Optional: manual test
    pipeline.manual_test()