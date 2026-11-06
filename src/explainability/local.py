"""Per-customer SHAP and LIME for serving: the batch scorer, ``/predict`` and ``/explain``.

Two rules keep these explanations honest:

* Explain the model in its own input space. The churn model is trained on scaled
  features (``churn_use_scaled`` on the model card), so it is explained on scaled
  inputs; feeding it raw values let large-unit features like ``monetary_total``
  dominate every explanation.
* Compare against a real reference. SHAP is measured against the average training
  customer (the scaler's mean); LIME samples around a set of real customers. Using
  the customer's own row as the reference made every attribution exactly zero.

Feature values are reported in the customer's own (raw) units.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _feature_cols(artifacts: Dict[str, Any]) -> List[str]:
    return list(artifacts["card"]["feature_cols"])


def model_inputs(artifacts: Dict[str, Any], X_raw: pd.DataFrame) -> pd.DataFrame:
    """Raw features -> the space the churn model was trained in."""
    cols = _feature_cols(artifacts)
    X = X_raw[cols].astype(float)
    if artifacts["card"].get("churn_use_scaled", True):
        return pd.DataFrame(artifacts["scaler"].transform(X), columns=cols, index=X.index)
    return X


def shap_drivers(artifacts: Dict[str, Any], X_raw: pd.DataFrame) -> List[List[Dict[str, Any]]]:
    """SHAP impact of every feature for each row, largest first, in log-odds of churn."""
    import shap

    cols = _feature_cols(artifacts)
    model = artifacts["churn_model"]
    X = model_inputs(artifacts, X_raw)
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:  # noqa: BLE001 - not a tree model
        average = pd.DataFrame([artifacts["scaler"].mean_], columns=cols)
        explainer = shap.LinearExplainer(model, model_inputs(artifacts, average))
    values = np.asarray(explainer.shap_values(X))
    if values.ndim == 3:
        values = values[:, :, -1]

    out = []
    for i in range(len(X)):
        row = [
            {
                "feature": col,
                "value": round(float(X_raw.iloc[i][col]), 4),
                "shap_impact": round(float(values[i, j]), 4),
                "direction": "Increases Churn" if values[i, j] > 0 else "Decreases Churn",
            }
            for j, col in enumerate(cols)
        ]
        out.append(sorted(row, key=lambda d: abs(d["shap_impact"]), reverse=True))
    return out


def lime_drivers(
    artifacts: Dict[str, Any],
    x_raw: pd.DataFrame,
    background_raw: pd.DataFrame,
    num_features: int = 8,
) -> List[Dict[str, Any]]:
    """LIME rules for one customer, read in raw units, sampled around real customers."""
    from lime.lime_tabular import LimeTabularExplainer

    cols = _feature_cols(artifacts)
    model = artifacts["churn_model"]

    def predict_proba(rows: np.ndarray) -> np.ndarray:
        return model.predict_proba(model_inputs(artifacts, pd.DataFrame(rows, columns=cols)))

    explainer = LimeTabularExplainer(
        training_data=background_raw[cols].to_numpy(dtype=float),
        feature_names=cols,
        class_names=["Retained", "Churned"],
        mode="classification",
        random_state=42,
    )
    exp = explainer.explain_instance(
        x_raw[cols].iloc[0].to_numpy(dtype=float), predict_proba, num_features=num_features
    )
    return [{"rule": rule, "weight": round(float(weight), 4)} for rule, weight in exp.as_list()]
