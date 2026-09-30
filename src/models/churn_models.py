from pathlib import Path
from typing import Dict, Any, List
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import xgboost as xgb
import lightgbm as lgb

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger
from src.models.data_split import split_data

logger = get_logger("churn_models")


def initialize_models(seed: int = 42) -> Dict[str, Any]:
    """
    Instantiates the 6 classical machine learning models specified in PRD Section 8.1.
    """
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"),
            "use_scaled": True,
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, random_state=seed, class_weight="balanced"),
            "use_scaled": False,
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=150, max_depth=8, min_samples_leaf=10, random_state=seed, class_weight="balanced", n_jobs=-1
            ),
            "use_scaled": False,
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1.0,
                random_state=seed,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            "use_scaled": False,
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                class_weight="balanced",
                verbosity=-1,
                n_jobs=-1,
            ),
            "use_scaled": False,
        },
        "KNN": {
            "model": KNeighborsClassifier(n_neighbors=7, weights="distance"),
            "use_scaled": True,
        },
    }
    return models


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cv_folds: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Trains model, runs cross-validation on train fold, evaluates on held-out test set,
    and computes all metrics required by PRD Section 11.
    """
    start_time = time.time()
    
    # 5-fold CV ROC-AUC on training partition
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    
    # Fit model on entire train set
    model.fit(X_train, y_train)
    train_time = round(time.time() - start_time, 3)

    # Predictions on held-out test set
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "test_roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "cv_roc_auc_mean": round(float(np.mean(cv_auc_scores)), 4),
        "cv_roc_auc_std": round(float(np.std(cv_auc_scores)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_time_sec": train_time,
    }
    return metrics


def train_and_compare_churn_models(save_artifacts: bool = True) -> pd.DataFrame:
    """
    Master function that trains all 6 classical churn models, serializes the artifacts,
    and produces a ranked comparison table.
    """
    logger.info("Initializing train/val/test data splits...")
    data = split_data()
    
    config = load_config()
    seed = config.get("project", {}).get("random_seed", 42)
    cv_folds = config.get("modeling", {}).get("cross_validation_folds", 5)

    models_dict = initialize_models(seed=seed)
    results = []

    root = get_project_root()
    art_dir = root / config.get("paths", {}).get("models_artifacts_dir", "models_artifacts")
    art_dir.mkdir(parents=True, exist_ok=True)

    best_auc = 0.0
    best_model_name = None

    for name, config_item in models_dict.items():
        logger.info(f"Training and evaluating: {name}...")
        model = config_item["model"]
        use_scaled = config_item["use_scaled"]

        X_tr = data["X_train_scaled"] if use_scaled else data["X_train"]
        X_te = data["X_test_scaled"] if use_scaled else data["X_test"]

        metrics = evaluate_model(
            model=model,
            X_train=X_tr,
            y_train=data["y_train"],
            X_test=X_te,
            y_test=data["y_test"],
            cv_folds=cv_folds,
            seed=seed,
        )

        results.append({
            "Model": name,
            "Test ROC-AUC": metrics["test_roc_auc"],
            "Test Recall": metrics["recall"],
            "Test Precision": metrics["precision"],
            "Test F1": metrics["f1_score"],
            "Test Accuracy": metrics["accuracy"],
            "CV ROC-AUC (Mean ± Std)": f"{metrics['cv_roc_auc_mean']} ± {metrics['cv_roc_auc_std']}",
            "Training Time (s)": metrics["train_time_sec"],
        })

        if save_artifacts:
            model_slug = name.lower().replace(" ", "_")
            artifact_path = art_dir / f"churn_{model_slug}.joblib"
            joblib.dump(model, artifact_path)
            logger.info(f"Saved artifact to {artifact_path.name}")

        if metrics["test_roc_auc"] > best_auc:
            best_auc = metrics["test_roc_auc"]
            best_model_name = name
            if save_artifacts:
                joblib.dump(model, art_dir / "best_churn_model.joblib")

    leaderboard = pd.DataFrame(results).sort_values(by=["Test ROC-AUC", "Test Recall"], ascending=False)
    leaderboard.reset_index(drop=True, inplace=True)

    # Save comparison table
    leaderboard_path = art_dir / "churn_model_comparison.parquet"
    leaderboard.to_parquet(leaderboard_path)
    leaderboard.to_csv(art_dir / "churn_model_comparison.csv", index=False)

    logger.info(f"Best churn model: {best_model_name} (Test ROC-AUC: {best_auc})")
    return leaderboard


if __name__ == "__main__":
    leaderboard = train_and_compare_churn_models()
    print("\n=================== CLASSICAL CHURN MODEL LEADERBOARD ===================")
    print(leaderboard.to_string(index=False))
