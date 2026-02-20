import numpy as np
import pandas as pd


class DummyModel:
    def __init__(self):
        self.feature_name_ = [
            "lag_1", "lag_7", "lag_14",
            "rolling_mean_7", "rolling_std_7",
            "month", "day_of_week", "quarter"
        ]

    def predict(self, X):
        return np.array([np.log1p(100)])


def test_prediction_inverse_transform():
    model = DummyModel()

    feature_row = pd.DataFrame([[0]*8], columns=model.feature_name_)

    pred_log = model.predict(feature_row)
    pred_actual = np.expm1(pred_log)

    assert round(pred_actual[0], 2) == 100.00