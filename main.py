# Pipeline entry point: load, engineer features, train, evaluate and save the best model

from src.data import prepare_dataset, split_features_target, split_train_test
from src.train import train_linear_regression, tune_random_forest, tune_lightgbm, evaluate_model, save_model
import config

def main():
    data_featured = prepare_dataset()
    print(f"Featured dataset shape: {data_featured.shape}")

    X, y = split_features_target(data_featured)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    lr_model = train_linear_regression(X_train, y_train)
    lr_rmse, _ = evaluate_model(lr_model, X_test, y_test)
    print(f"Linear Regression test RMSE: {lr_rmse:.2f}")

    rf_model, rf_params, rf_cv_rmse = tune_random_forest(X_train, y_train)
    rf_rmse, _ = evaluate_model(rf_model, X_test, y_test)
    print(f"Random Forest best params: {rf_params} | CV RMSE: {rf_cv_rmse:.2f} | test RMSE: {rf_rmse:.2f}")

    lgb_model, lgb_params, lgb_cv_rmse = tune_lightgbm(X_train, y_train)
    lgb_rmse, _ = evaluate_model(lgb_model, X_test, y_test)
    print(f"LightGBM best params: {lgb_params} | CV RMSE: {lgb_cv_rmse:.2f} | test RMSE: {lgb_rmse:.2f}")

    results = {
        'Linear Regression': (lr_model, lr_rmse),
        'Random Forest': (rf_model, rf_rmse),
        'LightGBM': (lgb_model, lgb_rmse)
    }
    best_name = min(results, key=lambda k: results[k][1])
    best_model, best_rmse = results[best_name]
    print(f"\nBest model: {best_name} (test RMSE: {best_rmse:.2f}, target: <= {config.RMSE_TARGET})")

    save_model(best_model, X_train.columns.tolist())

if __name__ == "__main__":
    main()
