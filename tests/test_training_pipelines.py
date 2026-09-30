import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from src.models import churn_models, clv_regressor


@pytest.fixture
def tiny_split(feature_table):
    from src.models.data_split import split_data

    return split_data(feature_table, save_scaler=False)


def test_initialize_models_contains_required_six():
    models = churn_models.initialize_models(seed=1)
    assert set(models) == {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "KNN",
    }
    for item in models.values():
        assert "model" in item and "use_scaled" in item


def test_evaluate_model_decision_function_branch(tiny_split):
    from sklearn.svm import LinearSVC

    clf = LinearSVC(max_iter=200)
    metrics = churn_models.evaluate_model(
        model=clf,
        X_train=tiny_split["X_train_scaled"],
        y_train=tiny_split["y_train"],
        X_test=tiny_split["X_test_scaled"],
        y_test=tiny_split["y_test"],
        cv_folds=2,
    )
    assert 0.0 <= metrics["test_roc_auc"] <= 1.0
    assert "confusion_matrix" in metrics


def test_train_and_compare_churn_models(monkeypatch, tmp_path, tiny_split):
    monkeypatch.setattr(churn_models, "split_data", lambda: tiny_split)
    monkeypatch.setattr(churn_models, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        churn_models,
        "initialize_models",
        lambda seed=42: {
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=200),
                "use_scaled": True,
            },
            "Decision Tree": {
                "model": churn_models.DecisionTreeClassifier(max_depth=3),
                "use_scaled": False,
            },
        },
    )

    leaderboard = churn_models.train_and_compare_churn_models(save_artifacts=True)

    assert isinstance(leaderboard, pd.DataFrame)
    assert len(leaderboard) == 2
    assert "Test ROC-AUC" in leaderboard.columns
    assert (tmp_path / "models_artifacts/churn_model_comparison.csv").exists()
    assert (tmp_path / "models_artifacts/best_churn_model.joblib").exists()
    # sorted descending by ROC-AUC
    assert leaderboard["Test ROC-AUC"].is_monotonic_decreasing


def test_initialize_regressors_contains_five():
    regressors = clv_regressor.initialize_regressors(seed=1)
    assert set(regressors) == {
        "Ridge Regression",
        "Random Forest Regressor",
        "Gradient Boosting Regressor",
        "XGBoost Regressor",
        "LightGBM Regressor",
    }


def test_evaluate_regressor_with_log_transform(tiny_split):
    metrics = clv_regressor.evaluate_regressor(
        model=Ridge(alpha=1.0),
        X_train=tiny_split["X_train_scaled"],
        y_train=tiny_split["clv_train"],
        X_test=tiny_split["X_test_scaled"],
        y_test=tiny_split["clv_test"],
        use_log_transform=True,
    )
    assert "r2_score" in metrics
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0


def test_train_and_compare_clv_regressors(monkeypatch, tmp_path, tiny_split):
    monkeypatch.setattr(clv_regressor, "split_data", lambda: tiny_split)
    monkeypatch.setattr(clv_regressor, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        clv_regressor,
        "initialize_regressors",
        lambda seed=42: {
            "Ridge Regression": {"model": Ridge(alpha=1.0), "use_scaled": True},
            "XGBoost Regressor": {
                "model": clv_regressor.xgb.XGBRegressor(n_estimators=10, max_depth=2),
                "use_scaled": False,
            },
        },
    )

    leaderboard = clv_regressor.train_and_compare_clv_regressors(save_artifacts=True)

    assert isinstance(leaderboard, pd.DataFrame)
    assert len(leaderboard) == 2
    assert {"Model", "R² Score", "MAE (£)"}.issubset(leaderboard.columns)
    assert (tmp_path / "models_artifacts/clv_model_comparison.csv").exists()
    assert (tmp_path / "models_artifacts/best_clv_model.joblib").exists()
