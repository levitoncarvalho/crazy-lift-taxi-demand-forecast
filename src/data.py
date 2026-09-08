# Data loading, resampling and feature engineering

import pandas as pd
from sklearn.model_selection import train_test_split
import config

def load_data(filepath=config.DATA_PATH):
    # Load the raw dataset with a datetime index
    data = pd.read_csv(filepath, index_col=[0], parse_dates=[0])
    data.sort_index(inplace=True)
    return data

def resample_hourly(data):
    # Resample to one-hour buckets, summing orders within each hour
    return data.resample('1h').sum()

def make_features(df, max_lag=config.MAX_LAG, rolling_mean_size=config.ROLLING_MEAN_SIZE):
    # Add calendar features, lag features and a leakage-safe rolling mean
    df_features = df.copy()

    df_features['dayofweek'] = df_features.index.dayofweek
    df_features['hour'] = df_features.index.hour

    for lag in range(1, max_lag + 1):
        df_features[f'lag_{lag}'] = df_features[config.TARGET_COL].shift(lag)

    # shift() before the rolling mean avoids leaking the current hour into its own feature
    df_features['rolling_mean'] = df_features[config.TARGET_COL].shift().rolling(rolling_mean_size).mean()

    return df_features

def prepare_dataset(filepath=config.DATA_PATH):
    # Full pipeline: load, resample, engineer features, drop warm-up rows
    data = load_data(filepath)
    data = resample_hourly(data)
    data_featured = make_features(data)
    data_featured = data_featured.dropna()
    return data_featured

def split_features_target(data_featured):
    # Split into features (X) and target (y)
    X = data_featured.drop(config.TARGET_COL, axis=1)
    y = data_featured[config.TARGET_COL]
    return X, y

def split_train_test(X, y, test_size=config.TEST_SIZE):
    # Chronological split (no shuffling) so the test set is the most recent period
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=test_size)
    return X_train, X_test, y_train, y_test
