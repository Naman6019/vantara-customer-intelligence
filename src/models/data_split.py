from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("data_split")

FEATURE_COLS = [
    "frequency",
    "total_items_bought",
    "unique_skus_bought",
    "monetary_total",
    "recency",
    "customer_age_days",
    "monetary_avg",
    "avg_basket_size",
    "historical_clv",
    "interpurchase_interval_mean",
    "interpurchase_interval_std",
    "purchase_frequency_trend",
    "return_rate",
    "cancellation_rate",
    "seasonal_concentration",
    "discount_sensitivity",
    "is_uk_customer",
    "engagement_score",
]


def load_processed_feature_table() -> pd.DataFrame:
    """Loads the customer-level processed feature table."""
    root = get_project_root()
    config = load_config()
    proc_path = root / config.get("paths", {}).get("processed_dir", "data/processed") / "customer_features.parquet"
    if not proc_path.exists():
        raise FileNotFoundError(f"Processed feature table not found at: {proc_path}. Run feature builder first.")
    return pd.read_parquet(proc_path)


def split_data(
    df: Optional[pd.DataFrame] = None,
    save_scaler: bool = True
) -> Dict[str, Any]:
    """
    Performs a 70/15/15 Stratified Train / Validation / Test split.
    Fits StandardScaler on training features only (preventing test data leakage)
    and saves the fitted scaler artifact.
    
    Returns:
        Dict containing raw and scaled X and y matrices for train, val, and test partitions,
        along with customer_id mappings.
    """
    if df is None:
        df = load_processed_feature_table()

    config = load_config()
    seed = config.get("project", {}).get("random_seed", 42)
    split_cfg = config.get("modeling", {}).get("train_val_test_split", {})
    train_ratio = split_cfg.get("train_ratio", 0.70)
    val_ratio = split_cfg.get("val_ratio", 0.15)
    test_ratio = split_cfg.get("test_ratio", 0.15)

    X = df[FEATURE_COLS].copy()
    y_churn = df["is_churned"].values
    y_clv = df["future_clv"].values
    customer_ids = df["customer_id"].values

    # Step 1: Split train (70%) vs temp (30%)
    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp, clv_train, clv_temp, id_train, id_temp = train_test_split(
        X, y_churn, y_clv, customer_ids,
        test_size=temp_ratio,
        random_state=seed,
        stratify=y_churn
    )

    # Step 2: Split temp equally into validation (15%) and test (15%)
    val_share_of_temp = val_ratio / temp_ratio
    X_val, X_test, y_val, y_test, clv_val, clv_test, id_val, id_test = train_test_split(
        X_temp, y_temp, clv_temp, id_temp,
        test_size=(1.0 - val_share_of_temp),
        random_state=seed,
        stratify=y_temp
    )

    # Step 3: Scale features - fit ONLY on X_train
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=FEATURE_COLS, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index
    )

    if save_scaler:
        root = get_project_root()
        art_dir = root / config.get("paths", {}).get("models_artifacts_dir", "models_artifacts")
        art_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = art_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Fitted StandardScaler persisted to: {scaler_path}")

    logger.info(
        f"Data split completed: Train={len(X_train):,} ({len(X_train)/len(X):.1%}), "
        f"Val={len(X_val):,} ({len(X_val)/len(X):.1%}), "
        f"Test={len(X_test):,} ({len(X_test)/len(X):.1%})"
    )
    logger.info(
        f"Churn class balance: Train={y_train.mean():.2%}, Val={y_val.mean():.2%}, Test={y_test.mean():.2%}"
    )

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "X_train_scaled": X_train_scaled, "X_val_scaled": X_val_scaled, "X_test_scaled": X_test_scaled,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "clv_train": clv_train, "clv_val": clv_val, "clv_test": clv_test,
        "id_train": id_train, "id_val": id_val, "id_test": id_test,
        "feature_cols": FEATURE_COLS,
    }


if __name__ == "__main__":
    split_data()
