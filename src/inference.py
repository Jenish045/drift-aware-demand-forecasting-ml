import joblib
import numpy as np
import pandas as pd
import os

# Load Model Artifacts

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "category_models.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model file not found at models/category_models.pkl. "
        "Please train models before running inference."
    )

category_models = joblib.load(MODEL_PATH)


# Utility: Build Feature Row in Correct Order

def build_feature_row(input_dict, feature_columns):

    missing_features = [f for f in feature_columns if f not in input_dict]

    if missing_features:
        raise ValueError(
            f"Missing features required by model: {missing_features}"
        )

    row = {col: input_dict[col] for col in feature_columns}
    return pd.DataFrame([row])


# Prediction Function

def predict(input_features: dict) -> float:

    # Input Validation

    if not isinstance(input_features, dict):
        raise TypeError("Input must be a dictionary.")

    required_features = [
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_std_7",
        "month",
        "day_of_week",
        "quarter"
    ]

    for feat in required_features:
        if feat not in input_features:
            raise ValueError(f"Missing required feature: {feat}")

    # Category Identification

    category_cols = [
        key for key in input_features
        if key.startswith("Product_Category_") and input_features[key] == 1
    ]

    if len(category_cols) != 1:
        raise ValueError(
            "Exactly one Product_Category dummy must be set to 1."
        )

    category_col = category_cols[0]

    if category_col not in category_models:
        raise ValueError(
            f"No trained model found for category: {category_col}"
        )

    model = category_models[category_col]

    if not hasattr(model, "feature_name_"):
        raise AttributeError("Model missing feature metadata.")

    feature_columns = model.feature_name_

    # Build Feature Row

    feature_row = build_feature_row(input_features, feature_columns)

    # Prediction

    pred_log = model.predict(feature_row)
    pred_actual = np.expm1(pred_log)

    # Enforce business constraint (no negative demand)
    final_prediction = max(0.0, float(pred_actual[0]))

    return final_prediction


# Example Usage

if __name__ == "__main__":

    # Pick any category model to test
    example_category = "Product_Category_Category_020"
    model = category_models[example_category]

    feature_columns = model.feature_name_

    # Start with all zeros
    sample_input = {feature: 0 for feature in feature_columns}

    # Populate realistic numeric values

    sample_input["lag_1"] = 25000
    sample_input["lag_7"] = 22000
    sample_input["lag_14"] = 27000

    sample_input["rolling_mean_7"] = 24000
    sample_input["rolling_std_7"] = 4500
    sample_input["rolling_max_7"] = 35000
    sample_input["rolling_std_14"] = 4800

    sample_input["growth_1"] = 0.08

    sample_input["month"] = 11
    sample_input["day_of_week"] = 5
    sample_input["quarter"] = 4
    sample_input["is_weekend"] = 1

    # Set exactly one warehouse dummy to 1

    for col in feature_columns:
        if col.startswith("Warehouse_"):
            sample_input[col] = 0

    warehouse_cols = [c for c in feature_columns if c.startswith("Warehouse_")]
    if warehouse_cols:
        sample_input[warehouse_cols[0]] = 1

    # Set exactly one category dummy to 1

    for col in feature_columns:
        if col.startswith("Product_Category_"):
            sample_input[col] = 0

    sample_input[example_category] = 1

    # Predict

    prediction = predict(sample_input)

    print("Testing category:", example_category)
    print("Predicted Demand:", prediction)