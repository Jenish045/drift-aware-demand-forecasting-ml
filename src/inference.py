import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "category_models.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Train models first.")

category_models = joblib.load(MODEL_PATH)


def build_feature_row(input_dict, feature_columns):
    row = {col: 0 for col in feature_columns}
    for key, value in input_dict.items():
        if key in row:
            row[key] = value
    return pd.DataFrame([row])


def predict(input_features: dict):

    required_features = [
        "lag_1", "lag_7", "lag_14",
        "rolling_mean_7", "rolling_std_7",
        "month", "day_of_week", "quarter"
    ]

    for feat in required_features:
        if feat not in input_features:
            raise ValueError(f"Missing required feature: {feat}")

    category_cols = [
        key for key in input_features
        if key.startswith("Product_Category_") and input_features[key] == 1
    ]

    if len(category_cols) != 1:
        raise ValueError("Exactly one Product_Category dummy must be set to 1.")

    category_col = category_cols[0]

    if category_col not in category_models:
        raise ValueError(f"No trained model found for {category_col}")

    model = category_models[category_col]
    feature_columns = model.feature_name_

    feature_row = build_feature_row(input_features, feature_columns)

    pred_log = model.predict(feature_row)
    pred_actual = np.expm1(pred_log)

    return float(pred_actual[0])


if __name__ == "__main__":

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

    prediction = predict(sample_input)
    print("Predicted Demand:", prediction)