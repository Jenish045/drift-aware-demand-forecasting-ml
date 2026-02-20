# Drift-Aware Retail Demand Forecasting

## Overview

This project implements a drift-aware retail demand forecasting system using segmented gradient boosting models.

Instead of building a simple regression model, the focus of this project is:

- Time-aware validation
- Architectural improvement
- Category-level segmentation
- Outlier control
- Drift monitoring
- Production-ready inference design

The final system is robust, stable under temporal shifts, and aligned with business evaluation metrics.

---

## Problem Statement

Retail demand forecasting is challenging due to:

- Heavy-tailed demand distributions
- Extreme spikes
- Intermittent demand patterns
- Category-level heterogeneity
- Temporal distribution drift

A single global model often fails to capture structural differences between product categories.

This project explores structural and architectural improvements to address these challenges.

---

## Dataset

Columns used:

- `Product_Code`
- `Warehouse`
- `Product_Category`
- `Date`
- `Order_Demand`

---

## Feature Engineering

### Lag Features
- lag_1
- lag_7
- lag_14

### Rolling Features
- rolling_mean_7
- rolling_std_7

### Calendar Features
- month
- day_of_week
- quarter

### Target Transformation
- log1p transformation
- expm1 inverse transform during inference

---

## Modeling Evolution

### 1. Baseline Global Model
- RandomForestRegressor
- Chronological train-test split
- Evaluated using MAE and RMSE

### 2. Temporal Cross-Validation
- TimeSeriesSplit (5 folds)
- Introduced WMAPE (Weighted Mean Absolute Percentage Error)
- Identified performance instability

### 3. Model Upgrade
- Replaced RandomForest with LightGBM
- Observed marginal algorithmic improvement

### 4. Category-Level Segmentation
- Trained separate LightGBM models per Product_Category
- Reduced cross-category variance
- Weighted evaluation across segments

### 5. Outlier Control
- Applied 99th percentile clipping per category
- Improved metric stability

### 6. Drift Analysis
- Compared train vs test mean demand
- Measured per-category WMAPE
- Verified absence of catastrophic category failure

---

## Final Architecture

- Segmented LightGBM models
- Log-transformed target
- TimeSeriesSplit validation
- Weighted WMAPE evaluation
- Outlier clipping
- Drift-aware evaluation
- Production-style model routing
- Artifact serialization using joblib

---

## Performance Summary

| Stage | Weighted WMAPE |
|--------|----------------|
| Global RandomForest | ~0.81 |
| Global LightGBM | ~0.79 |
| Category-Level LightGBM | ~0.74 |
| Category + Clipping | ~0.70 |

The final architecture significantly improved stability and reduced structural variance.

---

## Running Inference

### Ensure Model Exists

```
models/category_models.pkl
```

### Run

```
python src/inference.py
```

### Example Input

```python
sample_input = {
    "lag_1": 100,
    "lag_7": 120,
    "lag_14": 90,
    "rolling_mean_7": 110,
    "rolling_std_7": 15,
    "month": 6,
    "day_of_week": 2,
    "quarter": 2,
    "Product_Category_Category_020": 1
}
```

### Output

```
Predicted Demand: 109.52
```

---

## Unit Testing

Basic unit tests are included in the `tests/` directory to verify:

- WMAPE correctness
- Lag feature generation
- Inference inverse transformation

Run tests using:

```
pip install pytest
pytest
```

---

## Project Structure

```
project-root/
│
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_model_architecture.ipynb
│
├── src/
│   └── inference.py
│
├── tests/
│
├── data/
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
- PyTest

---

## Key Insights

- Architecture design matters more than algorithm switching.
- Time-aware validation is essential in forecasting.
- Heavy-tailed retail demand requires scale-normalized metrics.
- Category segmentation reduces structural instability.
- Drift monitoring improves real-world robustness.

---

## Future Improvements

- Hierarchical forecasting reconciliation
- Hyperparameter optimization (Optuna)
- External regressors (price, promotion, holidays)
- Automated drift detection (PSI)
- Deployment via FastAPI

---

## Author

Jenish Upadhyay