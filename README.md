# Drift-Aware Global Retail Demand Forecasting System

A production-oriented machine learning system for multi-entity retail demand forecasting with drift monitoring and expanding-window retraining simulation.

---

## 1. Problem Definition

We model demand forecasting as a supervised regression problem:

(Product_Code, Warehouse, Date) → Order_Demand

The objective is to estimate future demand for a specific product–warehouse pair using historical signals while maintaining time-aware validation and monitoring model stability over time.

This is implemented as a **global forecasting model** trained across all products and warehouses.

---

## 2. Dataset

Columns:

- Product_Code
- Warehouse
- Product_Category
- Date
- Order_Demand

Characteristics:

- 2160 products
- 4 warehouses
- 33 categories
- Heavy right-skew in target variable
- High variance and extreme outliers

Preprocessing:

- Date parsing and chronological sorting
- Numeric cleaning of Order_Demand
- Time-aware splitting (no random shuffling)
- Duplicate validation at (Product_Code, Warehouse, Date) level

---

## 3. Feature Engineering

### 3.1 Temporal Features
- day_of_week
- month
- quarter
- weekend indicator

### 3.2 Lag Features
- lag_1
- lag_7
- lag_14

### 3.3 Rolling Statistics
- rolling_mean_7
- rolling_std_7

### 3.4 Categorical Encoding
- One-hot encoding for Warehouse
- One-hot encoding for Product_Category
- Product_Code encoded in global feature space

The resulting feature matrix represents multi-entity time-series signals in tabular form.

---

## 4. Target Transformation

The target variable is highly skewed.

To stabilize variance and reduce the impact of extreme values:

y_log = log1p(Order_Demand)

Model predictions are converted back using:

Order_Demand = expm1(prediction_log)

This significantly improved MAE and reduced RMSE instability.

---

## 5. Modeling Strategy

### 5.1 Baselines
- Naive (last observation)
- Moving Average

### 5.2 ML Models
- Linear Regression
- Random Forest Regressor

### 5.3 Time-Aware Validation

Chronological 80/20 split used initially.

Further evaluation performed using expanding-window simulation:

- Train: 0–70%, Test: 70–80%
- Train: 0–80%, Test: 80–90%
- Train: 0–90%, Test: 90–100%

This mimics production retraining cycles.

---

## 6. Drift Monitoring

Three levels of monitoring were implemented:

### 6.1 Temporal Performance Tracking
- Monthly MAE computed over test period
- Checked for monotonic degradation

### 6.2 Target Distribution Comparison
- Train vs Test histogram comparison
- Mean and median stability analysis

### 6.3 Feature Distribution Drift
- rolling_mean_7
- lag_7
- lag_14

No structural drift detected, but expanding-window retraining improved stability.

---

## 7. Model Selection

Best performing configuration:

Random Forest + Log-Transformed Target

Key observations:

- Significant MAE reduction vs non-transformed model
- Lower RMSE volatility
- Feature importance dominated by rolling_mean_7 and lag features
- Demonstrated nonlinear dependency on recent demand signals

---

## 8. Model Persistence & Inference

Model saved locally using joblib.

Artifacts excluded from version control via .gitignore.

Inference pipeline:

- Loads trained model
- Constructs feature row for specific product/warehouse/date
- Applies log-space prediction
- Converts back to demand units
- Outputs business-readable forecast

Run:

python src/inference.py

---

## 9. Project Structure

project-root/

├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_global_ml_model.ipynb
│
├── src/
│   └── inference.py
│
├── models/              (local artifacts, ignored)
├── data/
│   └── raw/
│
├── requirements.txt
├── .gitignore
└── README.md

---

## 10. Engineering Practices Applied

- Strict chronological validation
- Feature leakage avoidance
- Artifact separation from source control
- Model version compatibility awareness
- Production-style inference script
- Expanding-window retraining simulation

---

## 11. Key Technical Takeaways

- Global tabular models can effectively handle multi-entity forecasting.
- Log transformation is critical for heavy-tailed retail demand.
- Drift detection requires both performance and distribution monitoring.
- Expanding-window retraining improves robustness even without strong drift signals.
- Proper Git hygiene is essential for ML projects.

---

## 12. Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

---

## Author

Jenish Upadhyay