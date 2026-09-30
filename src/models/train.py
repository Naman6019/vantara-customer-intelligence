"""Master training pipeline: tune -> select on validation -> calibrate -> test once.

Run with ``python -m src.models.train``. Produces:

  * ``models_artifacts/churn_best_model.joblib``   (production-refit classifier)
  * ``models_artifacts/clv_best_model.joblib``     (production-refit value model)
  * ``models_artifacts/churn_calibrator.joblib``   (isotonic calibrator)
  * ``models_artifacts/scaler.joblib``             (feature scaler)
  * ``models_artifacts/reference_distribution.json`` (drift reference)
  * ``models_artifacts/model_card.json``           (single source of truth)
  * MLflow runs under ``mlruns/``
"""

import json
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.models.calibration import ProbabilityCalibrator, choose_threshold
from src.models.churn_models import (
    evaluate_on_test,
    save_leaderboard,
    select_best,
    tune_and_compare_churn_models,
)
from src.models.clv_regressor import (
    evaluate_regressor,
    tune_and_compare_clv_regressors,
    _train_matrix,
)
from src.models.data_split import FEATURE_COLS, refit_data, split_data
from src.models.model_card import build_model_card, write_model_card
from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("train")


def _positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _feature_reference(df: pd.DataFrame) -> Dict[str, Dict[str, list]]:
    """Save per-feature decile breakpoints from the training snapshot for drift (D7)."""
    ref: Dict[str, Dict[str, list]] = {}
    for col in FEATURE_COLS:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            continue
        edges = np.quantile(values, np.linspace(0.0, 1.0, 11))
        edges = np.maximum.accumulate(edges)
        ref[col] = {"edges": edges.tolist()}
    return ref


def _mlflow(log_fn) -> None:  # pragma: no cover - thin wrapper around optional mlflow
    try:
        log_fn()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"MLflow logging skipped: {exc}")


def train_all(save: bool = True) -> Dict[str, Any]:
    """Run the full training flow and return a summary dict."""
    config = load_config()
    root = get_project_root()
    art_dir = root / config.get("paths", {}).get("models_artifacts_dir", "models_artifacts")
    art_dir.mkdir(parents=True, exist_ok=True)
    seed = config.get("project", {}).get("random_seed", 42)
    recall_target = config.get("modeling", {}).get("calibration", {}).get("recall_target", 0.70)
    decile = config.get("modeling", {}).get("value_capture_decile", 0.10)

    data = split_data(save_scaler=True)
    logger.info("Loaded stacked snapshots and built time-based split.")

    # ---------------------------------------------------------------- churn
    churn_lb, churn_fitted = tune_and_compare_churn_models(data)
    logger.info("\n" + churn_lb.drop(columns=["Best Params"]).to_string(index=False))
    save_leaderboard(churn_lb, art_dir)

    best_name = select_best(
        {row["Model"]: {"roc_auc": row["Val ROC-AUC"]} for _, row in churn_lb.iterrows()}
    )
    use_scaled = bool(churn_lb.loc[churn_lb["Model"] == best_name, "use_scaled"].iloc[0])
    logger.info(f"Selected churn model on validation: {best_name} (use_scaled={use_scaled})")

    # Calibrate on the validation snapshot.
    calibrator = ProbabilityCalibrator(method=config.get("modeling", {}).get("calibration", {}).get("method", "isotonic"))
    val_X = data["X_val_scaled"] if use_scaled else data["X_val"]
    val_prob = _positive_proba(churn_fitted[best_name], val_X)
    calibrator.fit(val_prob, data["y_val"])
    cal_val = calibrator.predict(val_prob)
    threshold_info = choose_threshold(data["y_val"], cal_val, recall_target)

    # Score the test snapshot exactly once.
    test_X = data["X_test_scaled"] if use_scaled else data["X_test"]
    test_metrics = evaluate_on_test(churn_fitted[best_name], test_X, data["y_test"])
    cal_test = calibrator.predict(test_metrics.pop("probabilities"))
    test_metrics["threshold"] = threshold_info["threshold"]
    test_metrics["calibrated_brier"] = round(float(brier_score_loss(data["y_test"], cal_test)), 4)
    logger.info(f"TEST (used once) {best_name}: {test_metrics}")

    # Production refit on all snapshots up to refit_end.
    refit = refit_data()
    refit_X = refit["X_scaled"] if use_scaled else refit["X"]
    production_churn = churn_fitted[best_name].__class__(**churn_fitted[best_name].get_params())
    production_churn.fit(refit_X, refit["y"])

    # ---------------------------------------------------------------- CLV
    clv_lb, clv_fitted = tune_and_compare_clv_regressors(data, decile=decile)
    logger.info("\n" + clv_lb.to_string(index=False))
    clv_best_name = clv_lb.loc[clv_lb["Val MAE (£)"].idxmin(), "Model"]
    logger.info(f"Selected value model on validation: {clv_best_name}")

    # Test the value model once.
    from src.models.clv_regressor import initialize_regressors

    clv_test_metrics = evaluate_regressor(
        clv_fitted[clv_best_name],
        _train_matrix(data, data["X_train"]),
        data["clv_train"],
        _train_matrix(data, data["X_test"]),
        data["clv_test"],
        use_log_transform=initialize_regressors()[clv_best_name].get("log_target", False),
        decile=decile,
    )

    # Refit the value model on all data up to refit_end.
    refit_clv_X = _train_matrix({"spend_last_90d": refit["aux"]}, refit["X"])
    production_clv = clv_fitted[clv_best_name].__class__(**clv_fitted[clv_best_name].get_params())
    y_refit = np.log1p(np.maximum(0, refit["clv"])) if initialize_regressors()[clv_best_name].get("log_target") else refit["clv"]
    production_clv.fit(refit_clv_X, y_refit)

    # ---------------------------------------------------------- persistence
    ref = _feature_reference(data["X_train"])
    if save:
        joblib.dump(production_churn, art_dir / "churn_best_model.joblib")
        joblib.dump(production_clv, art_dir / "clv_best_model.joblib")
        calibrator.save(art_dir / "churn_calibrator.joblib")
        with open(art_dir / "reference_distribution.json", "w", encoding="utf-8") as f:
            json.dump(ref, f)

    roles = data["roles"]
    # Value-tier quantiles for the recommendation engine. Many customers have a
    # zero next-90-day value (they churn), so tiers are cut on positive values to
    # keep "mid" meaningful.
    val_all = np.concatenate([data["clv_train"], data["clv_val"]])
    positives = val_all[val_all > 0]
    tier_source = positives if positives.size else val_all
    value_tier_quantiles = {
        "low": float(np.quantile(tier_source, 0.33)),
        "high": float(np.quantile(tier_source, 0.75)),
    }
    clv_flags = initialize_regressors()[clv_best_name]
    card = build_model_card(
        churn_model_name=best_name,
        churn_model_path="models_artifacts/churn_best_model.joblib",
        clv_model_name=clv_best_name,
        clv_model_path="models_artifacts/clv_best_model.joblib",
        feature_cols=FEATURE_COLS,
        churn_use_scaled=use_scaled,
        clv_use_scaled=clv_flags.get("use_scaled", False),
        clv_needs_aux=clv_flags.get("needs_aux", True),
        clv_log_target=clv_flags.get("log_target", False),
        value_tier_quantiles=value_tier_quantiles,
        train_cutoffs=[str(pd.Timestamp(c).date()) for c in roles["train"]],
        val_cutoff=str(pd.Timestamp(roles["val"][0]).date()) if roles["val"] else "",
        test_cutoff=str(pd.Timestamp(roles["test"][0]).date()) if roles["test"] else "",
        threshold=threshold_info["threshold"],
        calibrator_path="models_artifacts/churn_calibrator.joblib",
        validation_metrics={
            "roc_auc": float(churn_lb.loc[churn_lb["Model"] == best_name, "Val ROC-AUC"].iloc[0]),
            "recall_at_threshold": round(float(threshold_info["recall"]), 4),
            "precision_at_threshold": round(float(threshold_info["precision"]), 4),
            "f1_at_threshold": round(float(threshold_info["f1"]), 4),
            "calibrated_brier": round(float(brier_score_loss(data["y_val"], cal_val)), 4),
        },
        test_metrics={k: v for k, v in test_metrics.items() if k != "confusion_matrix"}
        | {"confusion_matrix": test_metrics.get("confusion_matrix")},
        clv_validation_metrics={
            "mae": float(clv_lb.loc[clv_lb["Model"] == clv_best_name, "Val MAE (£)"].iloc[0]),
            "r2": float(clv_lb.loc[clv_lb["Model"] == clv_best_name, "Val R² Score"].iloc[0]),
            "value_capture_top_decile": float(
                clv_lb.loc[clv_lb["Model"] == clv_best_name, "Top-Decile Value Capture"].iloc[0]
            ),
        },
        clv_test_metrics=clv_test_metrics,
        reference_distribution_path="models_artifacts/reference_distribution.json",
        value_capture_decile=decile,
    )
    if save:
        write_model_card(card)

    # ------------------------------------------------------------- MLflow
    def _log():
        import os

        import mlflow

        # MLflow's filesystem backend needs an explicit opt-in flag in newer versions.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(config.get("mlflow", {}).get("tracking_uri", "file:./mlruns"))
        mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "vantara-churn-clv"))
        for _, row in churn_lb.iterrows():
            with mlflow.start_run(run_name=f"churn::{row['Model']}"):
                mlflow.log_params(row["Best Params"])
                mlflow.log_metrics({
                    "val_roc_auc": float(row["Val ROC-AUC"]),
                    "val_recall": float(row["Val Recall"]),
                    "val_precision": float(row["Val Precision"]),
                    "cv_roc_auc_fwd": float(row["CV ROC-AUC (fwd-chain)"]),
                })
        for _, row in clv_lb.iterrows():
            with mlflow.start_run(run_name=f"clv::{row['Model']}"):
                mlflow.log_metrics({
                    "val_r2": float(row["Val R² Score"]),
                    "val_mae": float(row["Val MAE (£)"]),
                    "val_value_capture": float(row["Top-Decile Value Capture"]),
                })
        with mlflow.start_run(run_name="selected"):
            mlflow.log_param("churn_model", best_name)
            mlflow.log_param("clv_model", clv_best_name)
            mlflow.log_param("threshold", threshold_info["threshold"])
            mlflow.log_metrics({"test_roc_auc": test_metrics["roc_auc"], "test_recall": test_metrics["recall"]})
            mlflow.log_artifact(str(art_dir / "model_card.json"))
            mlflow.log_artifact(str(art_dir / "churn_model_comparison.csv"))

    _mlflow(_log)

    logger.info("Training complete. Model card written.")
    return {
        "churn_model": best_name,
        "clv_model": clv_best_name,
        "threshold": threshold_info["threshold"],
        "test_metrics": test_metrics,
        "clv_test_metrics": clv_test_metrics,
        "model_card": card,
    }


if __name__ == "__main__":
    summary = train_all()
    print(f"\nSelected churn model: {summary['churn_model']}")
    print(f"Selected value model: {summary['clv_model']}")
    print(f"Operating threshold: {summary['threshold']:.4f}")
    print(f"Test metrics: {summary['test_metrics']}")
