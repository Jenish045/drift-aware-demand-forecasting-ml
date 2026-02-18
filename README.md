# Drift-Aware Retail Demand Forecasting

## Overview

This project implements a drift-aware, segmented retail demand forecasting system using time-series validation and gradient boosting models.

The system evolves from a baseline global model into a category-level architecture with outlier control and drift monitoring, reflecting realistic retail forecasting practices.

---

## Problem Statement

Retail demand forecasting is challenging due to:

- Heavy-tailed demand distributions
- Extreme demand spikes
- Intermittent demand behavior
- Temporal distribution shifts
- Heterogeneous product dynamics

A global model often fails to capture category-specific demand patterns.

This project investigates architectural and statistical solutions to improve forecasting robustness.

---

## Dataset

Columns used:

- `Product_Code`
- `Warehouse`
- `Product_Category`
- `Date`
- `Order_Demand`

Feature engineering includes:

- Lag features (lag_1, lag_7, lag_14)
- Rolling statistics (rolling_mean_7, rolling_std_7)
- Calendar features (month, day_of_week, quarter)
- One-hot encoding for categorical variables

---

## Modeling Approach

### 1. Baseline Global Model
- RandomForestRegressor
- Time-aware train-test split
- Evaluated with MAE and RMSE

### 2. TimeSeries Cross-Validation
- 5-fold TimeSeriesSplit
- Introduced WMAPE (Weighted Mean Absolute Percentage Error)
- Identified instability under extreme demand volatility

### 3. Algorithm Upgrade
- Replaced RandomForest with LightGBM
- Minor improvements observed
- Concluded structural modeling issue

### 4. Category-Level Segmentation
- Trained separate LightGBM model per Product_Category
- Reduced cross-category variance
- Weighted evaluation across categories

### 5. Outlier Treatment
- Applied 99th percentile clipping per category
- Reduced spike distortion in error metrics
- Improved weighted WMAPE

### 6. Drift Analysis
- Per-category temporal split (80/20)
- Compared train vs test mean demand
- Verified model stability under demand shift

---

## Final Architecture

- Segmented LightGBM models
- Log-transformed target
- TimeSeriesSplit validation
- Weighted WMAPE evaluation
- Outlier clipping (Winsorization)
- Drift-aware performance monitoring
- Model persistence via joblib

---

## Performance Summary

| Stage | Weighted WMAPE |
|--------|----------------|
| Global RandomForest | ~0.81 |
| Global LightGBM | ~0.79 |
| Category-Level LightGBM | ~0.74 |
| Category + Clipping | ~0.70 |

No category exhibited catastrophic failure (WMAPE > 1.0).

---

## Key Insights

- Structural segmentation improves stability more than algorithm switching.
- Extreme demand spikes heavily distort scale-normalized metrics.
- Outlier control is essential in heavy-tailed retail data.
- Drift-aware evaluation provides realistic model assessment.
- Architecture design is critical in time-series forecasting systems.

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
├── data/
│   └── raw/
│
├── src/
│   └── inference.py
│
├── models/
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

---

## Future Improvements

- Hierarchical forecasting reconciliation
- Hyperparameter optimization
- Intermittent demand modeling (Croston-style methods)
- External regressors (price, promotion, holidays)
- Model monitoring automation

---

## Author

Jenish Upadhyay