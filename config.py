# Global Configuration

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / 'data' / 'taxi.csv'
MODEL_PATH = PROJECT_ROOT / 'models' / 'demand_model.joblib'

RANDOM_STATE = 42
TEST_SIZE = 0.1
CV_SPLITS = 3
N_ITER = 20

TARGET_COL = 'num_orders'
MAX_LAG = 24
ROLLING_MEAN_SIZE = 12
RMSE_TARGET = 48

# Random Forest hyperparameter search space
RF_PARAM_DIST = {
    'max_depth': [5, 10, 15, 20, None],
    'n_estimators': [50, 100, 150],
    'min_samples_split': [2, 5, 10]
}

# LightGBM hyperparameter search space
LGB_PARAM_DIST = {
    'max_depth': [3, 5, 10, -1],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200, 300],
    'num_leaves': [20, 31, 50, 70]
}
