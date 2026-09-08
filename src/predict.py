# Utilities to load the trained model and forecast the next hour's demand

import joblib
import numpy as np
import pandas as pd
import config

def load_model(model_path=config.MODEL_PATH):
    # Load the trained model and its expected feature order
    bundle = joblib.load(model_path)
    return bundle['model'], bundle['feature_columns']

def build_features_from_history(history, target_dayofweek, target_hour, max_lag=config.MAX_LAG, rolling_mean_size=config.ROLLING_MEAN_SIZE):
    # history: list of the last `max_lag` hourly order counts, oldest to newest
    if len(history) < max_lag:
        raise ValueError(f"Need at least {max_lag} hourly values, got {len(history)}")

    recent = history[-max_lag:]

    features = {'dayofweek': target_dayofweek, 'hour': target_hour}
    for lag in range(1, max_lag + 1):
        # lag_1 is the most recent hour, lag_24 the oldest
        features[f'lag_{lag}'] = recent[-lag]

    features['rolling_mean'] = np.mean(recent[-rolling_mean_size:])

    return features

def predict_next_hour(model, feature_columns, history, target_dayofweek, target_hour):
    # Predict the number of taxi orders expected in the next hour
    features = build_features_from_history(history, target_dayofweek, target_hour)
    X = pd.DataFrame([features])[feature_columns]
    prediction = model.predict(X)[0]
    return max(0, round(float(prediction)))
