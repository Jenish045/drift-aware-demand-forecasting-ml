import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "random_forest_log_model.pkl")

def load_model():
    return joblib.load(model_path)

def build_feature_row(model, product_code, warehouse,
                      lag_1, lag_7, lag_14,
                      rolling_mean_7, rolling_std_7,
                      date):

    feature_names = model.feature_names_in_
    row = pd.DataFrame(columns=feature_names)
    row.loc[0] = 0

    # Fill time features
    row["month"] = date.month
    row["day_of_week"] = date.weekday()
    row["quarter"] = (date.month - 1) // 3 + 1

    # Fill lag features
    if "lag_1" in row.columns:
        row["lag_1"] = lag_1
    if "lag_7" in row.columns:
        row["lag_7"] = lag_7
    if "lag_14" in row.columns:
        row["lag_14"] = lag_14
    if "rolling_mean_7" in row.columns:
        row["rolling_mean_7"] = rolling_mean_7
    if "rolling_std_7" in row.columns:
        row["rolling_std_7"] = rolling_std_7

    # One-hot encoded warehouse
    warehouse_col = f"Warehouse_{warehouse}"
    if warehouse_col in row.columns:
        row[warehouse_col] = 1

    # One-hot encoded product
    product_col = f"Product_Code_{product_code}"
    if product_col in row.columns:
        row[product_col] = 1

    return row

def forecast(model, feature_row):
    prediction_log = model.predict(feature_row)
    prediction = np.expm1(prediction_log)
    return prediction[0]

if __name__ == "__main__":
    model = load_model()

    # Example realistic scenario
    product = "Product_0001"
    warehouse = "Whse_A"
    forecast_date = datetime(2024, 3, 1)

    feature_row = build_feature_row(
        model=model,
        product_code=product,
        warehouse=warehouse,
        lag_1=1200,
        lag_7=1100,
        lag_14=1000,
        rolling_mean_7=1150,
        rolling_std_7=150,
        date=forecast_date
    )

    predicted_demand = forecast(model, feature_row)

    print("--------------------------------------------------")
    print("Retail Demand Forecast")
    print("--------------------------------------------------")
    print(f"Product: {product}")
    print(f"Warehouse: {warehouse}")
    print(f"Forecast Date: {forecast_date.date()}")
    print(f"Predicted Demand: {round(predicted_demand, 2)} units")
    print("--------------------------------------------------")