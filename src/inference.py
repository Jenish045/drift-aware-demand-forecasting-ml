import joblib
import numpy as np
import pandas as pd
import os


# Load Models

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "category_models.pkl")

category_models = joblib.load(MODEL_PATH)


# Utility: Build Feature Row

def build_feature_row(input_dict, feature_columns):
    """
    Builds a single-row DataFrame matching training feature order.
    """
    row = {col: 0 for col in feature_columns}

    for key, value in input_dict.items():
        if key in row:
            row[key] = value

    return pd.DataFrame([row])


# Prediction Function

def predict(input_features: dict):
    """
    input_features must include:
        - Product_Category dummy column (e.g., Product_Category_Category_020)
        - lag features
        - rolling features
        - calendar features
    """

    # Identify category column
    category_col = None
    for key in input_features.keys():
        if key.startswith("Product_Category_") and input_features[key] == 1:
            category_col = key
            break

    if category_col is None:
        raise ValueError("No valid Product_Category dummy provided.")

    if category_col not in category_models:
        raise ValueError(f"No trained model found for {category_col}")

    model = category_models[category_col]

    feature_columns = model.feature_name_

    feature_row = build_feature_row(input_features, feature_columns)

    pred_log = model.predict(feature_row)
    pred_actual = np.expm1(pred_log)

    return float(pred_actual[0])


# Example Usage

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