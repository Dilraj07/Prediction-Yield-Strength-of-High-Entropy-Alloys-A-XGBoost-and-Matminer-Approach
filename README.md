# Prediction of Yield Strength in High-Entropy Alloys (HEA)
A Physics-Informed Machine Learning approach to predict mechanical properties of High-Entropy Alloys using Gradient Boosting and Domain Knowledge.

## 🚀 HEA Discovery Engine (Streamlit App)
This project includes a state-of-the-art interactive web application for material scientists to design and analyze new alloys.

### Feature Highlights
- **🧪 Alloy Composer**: Design new alloys and simulate their Yield Strength in real-time.
- **⚙️ Processing Simulator**: Adjust Temperature, Grain Size, and Manufacturing Method to see property shifts.
- **🧠 Explainable AI (SHAP)**: Understand *why* the model makes a prediction. Visualize the impact of Lattice Distortion (δ), VEC, and more.
- **📊 Dataset Analytics**: Interactive dashboard to explore correlations and property distributions in the training data.

![App Screenshot](https://via.placeholder.com/800x400?text=HEA+Discovery+Engine+Dashboard)

## 🛠️ Installation & Usage

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Dilraj07/Prediction-Yield-Strength-of-High-Entropy-Alloys-A-XGBoost-and-Matminer-Approach.git
   cd Prediction-Yield-Strength-of-High-Entropy-Alloys-A-XGBoost-and-Matminer-Approach
   ```

2. **Set up Environment**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
- `app.py`: Main Streamlit application file.
- `hea_prediction.py`: Core machine learning pipeline (Data Preprocessing, Feature Engineering, Model Training).
- `requirements.txt`: Python dependencies.
- `MPEA_dataset.csv`: Training dataset containing alloy compositions and properties.

## 🔬 Scientific Approach
The model integrates physical parameters known to influence solid solution strengthening:
- **Valence Electron Concentration (VEC)**: Predicts phase stability (FCC vs BCC).
- **Atomic Size Mismatch (δ)**: Quantifies lattice distortion.
- **Mixing Entropy (S_mix)**: Thermodynamics of multi-component systems.
- **Processing History**: Accounts for grain size and heat treatment effects.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements.
