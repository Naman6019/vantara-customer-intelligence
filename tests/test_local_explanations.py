import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.explainability.local import lime_drivers, shap_drivers

FEATURES = ["big_units", "small_units"]


def _toy_artifacts():
    """A scaled logistic model where the small-unit feature drives churn and the big-unit one barely does."""
    rng = np.random.RandomState(0)
    n = 600
    X = pd.DataFrame({
        "big_units": rng.normal(10_000, 3_000, n),  # e.g. £ spent: huge numbers, weak signal
        "small_units": rng.normal(0.5, 0.1, n),  # e.g. a rate: tiny numbers, strong signal
    })
    logit = 0.2 * (X["big_units"] - 10_000) / 3_000 + 3.0 * (X["small_units"] - 0.5) / 0.1
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression().fit(scaler.transform(X), y)
    artifacts = {
        "churn_model": model,
        "scaler": scaler,
        "card": {"feature_cols": FEATURES, "churn_use_scaled": True},
    }
    return artifacts, X


def test_single_customer_gets_non_zero_attributions():
    artifacts, X = _toy_artifacts()
    impacts = shap_drivers(artifacts, X.iloc[[0]])[0]
    assert any(abs(d["shap_impact"]) > 0 for d in impacts)


def test_attributions_add_up_to_the_models_log_odds_versus_the_average_customer():
    artifacts, X = _toy_artifacts()
    model, scaler = artifacts["churn_model"], artifacts["scaler"]
    row = X.iloc[[5]]
    impacts = shap_drivers(artifacts, row)[0]

    margin = model.decision_function(scaler.transform(row))[0]
    average = model.decision_function(scaler.transform(pd.DataFrame([scaler.mean_], columns=FEATURES)))[0]
    assert sum(d["shap_impact"] for d in impacts) == pytest.approx(margin - average, abs=1e-3)


def test_raw_units_do_not_decide_the_top_driver():
    artifacts, _ = _toy_artifacts()
    # Average spend, but an unusually high rate: the rate is what drives this customer's risk.
    customer = pd.DataFrame([{"big_units": 10_500.0, "small_units": 0.75}])
    impacts = shap_drivers(artifacts, customer)[0]
    assert impacts[0]["feature"] == "small_units"
    assert impacts[0]["value"] == 0.75  # shown in the customer's own units


def test_lime_weights_are_non_zero_and_find_the_real_driver():
    artifacts, X = _toy_artifacts()
    customer = pd.DataFrame([{"big_units": 10_500.0, "small_units": 0.75}])
    rules = lime_drivers(artifacts, customer, background_raw=X, num_features=2)
    assert any(abs(r["weight"]) > 0 for r in rules)
    assert "small_units" in max(rules, key=lambda r: abs(r["weight"]))["rule"]
