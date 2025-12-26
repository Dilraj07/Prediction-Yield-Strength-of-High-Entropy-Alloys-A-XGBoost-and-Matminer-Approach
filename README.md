# Yield Strength Prediction of High Entropy Alloys (HEAs)

This project utilizes machine learning (XGBoost) and physics-based feature engineering to predict the yield strength of High Entropy Alloys (HEAs) based on their chemical composition and processing conditions.

## Project Overview

High Entropy Alloys are a class of materials with superior mechanical properties. This tool helps researchers and material scientists predict the mechanical performance of new alloy compositions without expensive physical testing.

### Key Features
- **Physics-Informed Features**: Calculates Valency Electron Concentration (VEC), atomic radius difference ($\delta$), mixing entropy ($S_{mix}$), and electronegativity differences.
- **Machine Learning Model**: Uses a Gradient Boosting Regressor (XGBoost equivalent) for high-accuracy predictions.
- **Interactive Prediction**: Includes a CLI tool to input new alloy formulas and get instant yield strength estimates.
- **Feature Importance Analysis**: Visualizes which physical properties drive the material strength.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Dilraj07/Prediction-Yield-Strength-of-High-Entropy-Alloys-A-XGBoost-and-Matminer-Approach.git
   cd Prediction-Yield-Strength-of-High-Entropy-Alloys-A-XGBoost-and-Matminer-Approach
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the Model and Predict
Run the main script to train the model on the provided dataset (`MPEA_dataset.csv`) and start the interactive prediction session.

```bash
python hea_prediction.py
```

### 2. Feature Importance
The script automatically generates a `feature_importance.png` plot showing the most influential factors in determining yield strength.

## Dataset
The model is trained on `MPEA_dataset.csv`, which contains experimental data on various HEA compositions, processing methods, and measured yield strengths.

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
