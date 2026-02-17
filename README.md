# Demand Forecasting with Drift Detection & Retraining Strategy

## 📌 Project Overview

This project builds a **production-aware demand forecasting system** using historical product demand data.

Unlike typical forecasting notebooks, this project goes beyond basic modeling and includes:

- Structured feature engineering for time-series data  
- Statistical baselines  
- Machine learning models  
- Target transformation for skewed distributions  
- Model comparison and evaluation  
- Drift detection analysis  
- Expanding-window retraining simulation  

The objective is to simulate how real-world forecasting systems behave in production environments.

---

## 📊 Dataset Description

The dataset contains historical demand records with the following columns:

- `Product_Code`  
- `Warehouse`  
- `Product_Category`  
- `Date`  
- `Order_Demand`  

### Data Characteristics

- Highly right-skewed target variable  
- Extreme demand spikes  
- Large variance across products  
- Multiple warehouses and product categories  

---

## 🔍 Key Challenges

1. Heavy right-skewed demand distribution  
2. High variance and outliers  
3. Multi-entity forecasting (products + warehouses)  
4. Potential temporal performance degradation  

---

## ⚙️ Feature Engineering

### Temporal Features
- Day of week  
- Month  
- Quarter  
- Weekend indicator  

### Lag Features
- `lag_1`  
- `lag_7`  
- `lag_14`  

### Rolling Features
- `rolling_mean_7`  
- `rolling_std_7`  

### Categorical Encoding
- One-hot encoding for `Warehouse`  
- One-hot encoding for `Product_Category`  
- `Product_Code` excluded to avoid extreme dimensionality  

---

## 📈 Modeling Approach

### 1️⃣ Baseline Models
- Naive Forecast  
- Moving Average  

### 2️⃣ Machine Learning Models
- Linear Regression  
- Random Forest Regressor  

### 3️⃣ Target Transformation

Due to extreme skewness in `Order_Demand`, a log transformation was applied:

y_log = log(1 + y)

Models were trained on the transformed scale and predictions were converted back using `expm1` for evaluation.

---

## 🏆 Model Performance Summary

Best performing model:

**Random Forest + Log Transformation**

Improvements observed:
- Significant MAE reduction  
- More stable RMSE  
- Better handling of extreme demand spikes  

This demonstrates the importance of:
- Understanding target distribution  
- Applying variance stabilization techniques  
- Iterative modeling refinement  

---

## 📉 Drift Detection & Monitoring

This project includes production-style model monitoring.

### 1️⃣ Temporal Performance Monitoring

- Monthly MAE calculated over the test period  
- Checked for monotonic degradation  
- No consistent performance drift observed  

### 2️⃣ Distribution Drift Analysis

Compared training vs testing distributions for:

- Target (`Order_Demand`)  
- `rolling_mean_7`  
- `lag_7`  

Findings:
- No significant structural distribution shift  
- Feature distributions remained stable  

### 3️⃣ Expanding-Window Retraining Simulation

Simulated periodic retraining:

- Train on 70%, test next 10%  
- Retrain on 80%, test next 10%  

Retraining significantly reduced MAE in later windows, demonstrating that scheduled retraining improves forecasting stability even without severe drift.

---

## 🧠 Key Insights

- Demand data is highly skewed and volatile.  
- Log transformation is critical for stability.  
- Feature engineering drives performance improvement.  
- Drift detection is essential in production systems.  
- Expanding-window retraining enhances robustness.  

---

## 🛠 Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

## 📂 Project Structure

project-root/
│
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_global_ml_model.ipynb
│
├── data/
│   └── raw/
│
└── README.md

---

## 🚀 Future Improvements

- Hyperparameter tuning with cross-validation  
- Time-series cross-validation  
- Model persistence using joblib  
- Inference pipeline script  
- REST API deployment (FastAPI / Flask)  
- Automated retraining scheduler  

---

## 👤 Author

Jenish Upadhyay  
Machine Learning & Data Science Enthusiast  