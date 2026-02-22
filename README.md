# Drift-Aware Retail Demand Forecasting  
End-to-End Production-Grade ML System

---

## Overview

This project implements a **drift-aware, category-segmented retail demand forecasting system** designed to handle real-world retail volatility, category heterogeneity, and temporal distribution shift.

Instead of training a single global model, this system evolves into a:

- Time-aware
- Category-segmented
- Volatility-aware
- Drift-monitored
- Production-ready inference pipeline

The final architecture achieves:

> **6.6% Weighted WMAPE**

while maintaining stability across time-based validation splits.

---

## Problem Statement

Retail demand forecasting presents multiple challenges:

- Heavy-tailed demand distribution
- Extreme spikes and volatility
- Category-level structural differences
- Temporal drift
- Scale imbalance across products
- Risk of data leakage in naive splits

A single global model often fails to generalize across categories and time.

This project focuses on architectural corrections rather than simple algorithm replacement.

---

## Dataset

Columns used:

- `Product_Code`
- `Warehouse`
- `Product_Category`
- `Date`
- `Order_Demand`

Daily demand observations across multiple product categories and warehouses.

---

## Feature Engineering

### Lag Features
- lag_1
- lag_7
- lag_14

### Rolling Statistics
- rolling_mean_7
- rolling_std_7
- rolling_max_7
- rolling_std_14

### Volatility & Growth
- growth_1

### Calendar Features
- month
- day_of_week
- quarter
- is_weekend

### Encoding
- One-hot encoding for Product_Category
- One-hot encoding for Warehouse

### Target Transformation
- log1p transformation
- expm1 inverse transform during inference
- Stabilized heavy-tailed distribution

---

## Modeling Evolution

### 1️⃣ Global RandomForest
- Chronological split
- High instability
- Weighted WMAPE ≈ 81%

---

### 2️⃣ TimeSeriesSplit Validation
- Eliminated leakage
- Introduced business-aligned metric (WMAPE)
- Exposed structural instability

---

### 3️⃣ Global LightGBM
- Algorithmic improvement
- Minor gain
- WMAPE ≈ 79%

---

### 4️⃣ Category-Level Segmentation
- Separate LightGBM per Product_Category
- Reduced cross-category interference
- WMAPE ≈ 74%

---

### 5️⃣ Outlier Control (Per Category)
- 99th percentile clipping
- Reduced spike distortion
- WMAPE ≈ 70%

---

### 6️⃣ Volatility-Aware Features + Drift Monitoring (Final Architecture)
- Added rolling_max, rolling_std_14, growth features
- Improved spike sensitivity
- Residual bias reduced
- Stable time-split performance

Final Weighted WMAPE:

> **0.066 (6.6%)**

---

## Final Architecture

- Segmented LightGBM models (per category)
- TimeSeriesSplit validation
- Log-transformed target
- Per-category outlier clipping
- Volatility-aware features
- Drift monitoring (mean shift + WMAPE)
- Weighted evaluation
- Joblib model serialization
- Production-style inference routing

---

## Performance Summary

| Stage | Weighted WMAPE |
|--------|----------------|
| Global RandomForest | 0.81 |
| Global LightGBM | 0.79 |
| Category-Level LightGBM | 0.74 |
| Category + Clipping | 0.70 |
| Final Drift-Aware Architecture | **0.066** |

Residual Mean ≈ 0  
Stable across time splits  
No catastrophic category failures  

---

## Inference Pipeline

Models stored in:

```
models/category_models.pkl
```

Run:

```
python src/inference.py
```

Inference design:

- Validates required features
- Ensures exactly one category dummy
- Routes to correct category model
- Applies inverse log transformation
- Prevents negative outputs

Example input:

```python
sample_input = {
    "lag_1": 25000,
    "lag_7": 22000,
    "lag_14": 27000,
    "rolling_mean_7": 24000,
    "rolling_std_7": 4500,
    "rolling_max_7": 35000,
    "rolling_std_14": 4800,
    "growth_1": 0.08,
    "month": 11,
    "day_of_week": 5,
    "quarter": 4,
    "is_weekend": 1,
    "Warehouse_Whse_C": 1,
    "Product_Category_Category_020": 1
}
```

Output:

```
Predicted Demand: 1383.417578962056
```

---

## Drift Monitoring

Implemented:

- Train vs test mean comparison
- Per-category WMAPE analysis
- Residual distribution check
- Weighted performance tracking

Confirmed no structural category collapse.

---

## Project Structure

```
project-root/
│
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_global_ml_model.ipynb
│
├── src/
│   └── inference.py
│
├── tests/
│
├── models/
│   └── category_models.pkl
│
├── data/
│
└── README.md
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- Joblib
- PyTest

---

## Key Technical Learnings

- Architecture > algorithm swap
- Time-aware validation is mandatory in forecasting
- Weighted metrics reflect business better than MAE
- Segmentation reduces structural bias
- Volatility features dramatically improve spike modeling
- Drift monitoring is essential for production readiness

---

## Future Enhancements

- Hierarchical reconciliation
- Optuna hyperparameter tuning
- PSI-based automated drift detection
- FastAPI deployment
- Batch inference service
- Monitoring dashboard

---

## Author

Jenish Upadhyay  
Machine Learning | Forecasting Systems | Production ML Architecture