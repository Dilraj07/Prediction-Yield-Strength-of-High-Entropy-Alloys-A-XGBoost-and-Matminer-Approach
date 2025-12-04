# 🌟 Predictive Modeling of Yield Strength in High-Entropy Alloys (HEAs)

This project builds a robust and interpretable machine learning model to **predict the Yield Strength (σᵧ)** of **High-Entropy Alloys (HEAs)** using their chemical composition and fundamental physicochemical descriptors.  
It aims to accelerate materials discovery by enabling **fast virtual screening** of promising alloy compositions.

---

## 🔬 **Project Overview**

The workflow integrates modern **materials informatics** tools:

- **Matminer** → Feature engineering from HEA compositions  
- **XGBoost** → High-accuracy regression model for σᵧ prediction  

The complete pipeline transforms raw chemical formulas into high-dimensional features and trains a state-of-the-art ML model.

---

## 🧱 **Data Flow**

Raw Data -> Clean & Merge -> Matminer Featurization (X) -> XGBoost Model Training -> Yield Strength (σᵧ) Prediction


### **1. Create and Activate Virtual Environment**
```bash
python -m venv hea_stable
source hea_stable/bin/activate   # Linux/Mac
hea_stable\Scripts\activate      # Windows
```
model.py
│
├── Data Loading & Cleaning
│     ├── load CSVs
│     ├── remove missing σᵧ / temperature rows
│
├── Feature Engineering
│     ├── StrToComposition()
│     ├── ElementProperty()
│
└── Model Training & Evaluation
      ├── XGBRegressor()
      ├── Metrics (R², RMSE, MAE)
      └── Feature Importance Extraction
