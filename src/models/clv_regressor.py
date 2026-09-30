from pathlib import Path
from typing import Dict, Any, List
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger
from src.models.data_split import split_data

logger = get_logger("clv_regressor")


def initialize_regressors(seed: int = 42) -> Dict[str, Any]:
    """
    Initializes regressors for Customer Lifetime Value (future monetary spend) estimation.
    """
    models = {
        "Ridge Regression": {
            "model": Ridge(alpha=100.0, random_state=seed),
            "use_scaled": True,
        },
        "Random Forest Regressor": {
            "model": RandomForestRegressor(
                n_estimators=150, max_depth=8, min_samples_leaf=10, random_state=seed, n_jobs=-1
            ),
            "use_scaled": False,
        },
        "Gradient Boosting Regressor": {
            "model": GradientBoostingRegressor(
                n_estimators=150, max_depth=4, learning_rate=0.05, random_state=seed
            ),
            "use_scaled": False,
        },
        "XGBoost Regressor": {
            "model": xgb.XGBRegressor(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=-1,
            ),
            "use_scaled": False,
        },
        "LightGBM Regressor": {
            "model": lgb.LGBMRegressor(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                verbosity=-1,
                n_jobs=-1,
            ),
            "use_scaled": False,
        },
    }
    return models


def evaluate_regressor(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    use_log_transform: bool = False,
) -> Dict[str, Any]:
    """
    Trains regressor and evaluates R², MAE, and RMSE on test partition.
    """
    start_time = time.time()
    
    # Optional log1p target transformation to tame extreme right-tail skew
    if use_log_transform:
        y_train_fit = np.log1p(np.maximum(0, y_train))
    else:
        y_train_fit = y_train

    model.fit(X_train, y_train_fit)
    train_time = round(time.time() - start_time, 3)

    raw_preds = model.predict(X_test)
    if use_log_transform:
        y_pred = np.expm1(raw_preds)
    else:
        y_pred = raw_preds

    # Non-negative spend constraint
    y_pred = np.maximum(0, y_pred)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        "r2_score": round(float(r2), 4),
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "train_time_sec": train_time,
    }


def train_and_compare_clv_regressors(save_artifacts: bool = True) -> pd.DataFrame:
    """
    Trains and compares multiple regression models for future Customer Lifetime Value.
    """
    logger.info("Initializing train/val/test data splits for CLV regression...")
    data = split_data()
    
    config = load_config()
    seed = config.get("project", {}).get("random_seed", 42)

    regressors_dict = initialize_regressors(seed=seed)
    results = []

    root = get_project_root()
    art_dir = root / config.get("paths", {}).get("models_artifacts_dir", "models_artifacts")
    art_dir.mkdir(parents=True, exist_ok=True)

    best_r2 = -float("inf")
    best_model_name = None

    for name, config_item in regressors_dict.items():
        logger.info(f"Training CLV Regressor: {name}...")
        model = config_item["model"]
        use_scaled = config_item["use_scaled"]

        X_tr = data["X_train_scaled"] if use_scaled else data["X_train"]
        X_te = data["X_test_scaled"] if use_scaled else data["X_test"]

        metrics = evaluate_regressor(
            model=model,
            X_train=X_tr,
            y_train=data["clv_train"],
            X_test=X_te,
            y_test=data["clv_test"],
            use_log_transform=False,
        )

        results.append({
            "Model": name,
            "R² Score": metrics["r2_score"],
            "MAE (£)": metrics["mae"],
            "RMSE (£)": metrics["rmse"],
            "Training Time (s)": metrics["train_time_sec"],
        })

        if save_artifacts:
            model_slug = name.lower().replace(" ", "_")
            artifact_path = art_dir / f"clv_{model_slug}.joblib"
            joblib.dump(model, artifact_path)

        if metrics["r2_score"] > best_r2:
            best_r2 = metrics["r2_score"]
            best_model_name = name
            if save_artifacts:
                joblib.dump(model, art_dir / "best_clv_model.joblib")

    leaderboard = pd.DataFrame(results).sort_values(by="R² Score", ascending=False)
    leaderboard.reset_index(drop=True, inplace=True)

    # Save comparison table
    leaderboard.to_parquet(art_dir / "clv_model_comparison.parquet")
    leaderboard.to_csv(art_dir / "clv_model_comparison.csv", index=False)

    logger.info(f"Best CLV Regressor: {best_model_name} (R² Score: {best_r2})")
    return leaderboard


if __name__ == "__main__":
    clv_leaderboard = train_and_compare_clv_regressors()
    print("\n=================== CLV REGRESSION MODEL LEADERBOARD ===================")
    print(clv_leaderboard.to_string(index=False))
