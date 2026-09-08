# Model training and hyperparameter tuning

import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import config

def train_linear_regression(X_train, y_train):
    # Baseline model, no tuning needed
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def tune_random_forest(X_train, y_train):
    # Search for the best Random Forest hyperparameters with time-series-safe CV
    tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
    rf_base = RandomForestRegressor(random_state=config.RANDOM_STATE)

    rscv = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=config.RF_PARAM_DIST,
        n_iter=config.N_ITER,
        scoring='neg_root_mean_squared_error',
        cv=tscv,
        n_jobs=-1,
        random_state=config.RANDOM_STATE
    )
    rscv.fit(X_train, y_train)
    return rscv.best_estimator_, rscv.best_params_, -rscv.best_score_

def tune_lightgbm(X_train, y_train):
    # Search for the best LightGBM hyperparameters with time-series-safe CV
    tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
    lgb_base = LGBMRegressor(random_state=config.RANDOM_STATE, verbosity=-1)

    rscv = RandomizedSearchCV(
        estimator=lgb_base,
        param_distributions=config.LGB_PARAM_DIST,
        n_iter=config.N_ITER,
        scoring='neg_root_mean_squared_error',
        cv=tscv,
        n_jobs=-1,
        random_state=config.RANDOM_STATE
    )
    rscv.fit(X_train, y_train)
    return rscv.best_estimator_, rscv.best_params_, -rscv.best_score_

def evaluate_model(model, X_test, y_test):
    # Compute RMSE on the test set
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    return rmse, predictions

def save_model(model, feature_columns, model_path=config.MODEL_PATH):
    # Persist the trained model together with the expected feature column order
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'feature_columns': feature_columns}, model_path)
    print(f"Model saved to: {model_path}")
