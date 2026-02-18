import joblib
import numpy as np

def load_model():
    return joblib.load("../models/random_forest_log_model.pkl")

def predict(model, input_features):
    prediction_log = model.predict([input_features])
    prediction = np.expm1(prediction_log)
    return prediction[0]


if __name__ == "__main__":
    model = load_model()

    # Example feature vector (must match training feature order)
    sample_input = [0] * len(model.feature_importances_)

    prediction = predict(model, sample_input)
    print("Predicted Demand:", prediction)