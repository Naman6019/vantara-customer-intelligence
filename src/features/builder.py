from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("feature_builder")


def extract_temporal_windows(
    df: pd.DataFrame, cutoff_date: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits transaction logs strictly into historical observation window and future prediction window.
    Guarantees no target leakage by hard temporal boundary.
    """
    df_obs = df[df["invoice_date"] < cutoff_date].copy()
    df_pred = df[df["invoice_date"] >= cutoff_date].copy()
    
    logger.info(
        f"Temporal Split at {cutoff_date}: "
        f"Observation records: {len(df_obs):,}, Prediction records: {len(df_pred):,}"
    )
    return df_obs, df_pred


def compute_targets(
    df_obs: pd.DataFrame, df_pred: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes churn label (is_churned) and future CLV (future_clv) for all customers
    who were active before the cutoff date.
    
    A customer is considered churned (is_churned = 1) if they make ZERO valid positive purchases
    in the prediction window.
    """
    active_customers = sorted(df_obs["customer_id"].dropna().unique())
    
    # Positive purchases in prediction window
    pred_positive = df_pred[(df_pred["quantity"] > 0) & (~df_pred["is_cancellation"])]
    
    # Aggregated future spend in prediction window
    pred_spend = (
        df_pred.groupby("customer_id")["line_total"].sum().rename("future_clv")
    )
    
    pred_invoices = (
        pred_positive.groupby("customer_id")["invoice"].nunique().rename("future_invoices")
    )

    targets_df = pd.DataFrame({"customer_id": active_customers})
    targets_df = targets_df.merge(pred_spend, on="customer_id", how="left").fillna({"future_clv": 0.0})
    targets_df = targets_df.merge(pred_invoices, on="customer_id", how="left").fillna({"future_invoices": 0})
    
    # If customer placed 0 orders in prediction window, they churned
    targets_df["is_churned"] = (targets_df["future_invoices"] == 0).astype(int)
    # Clip negative future CLV if any to zero (returns with no purchases)
    targets_df["future_clv"] = targets_df["future_clv"].clip(lower=0.0)
    
    targets_df.drop(columns=["future_invoices"], inplace=True)
    logger.info(
        f"Target calculation complete: {len(targets_df):,} customers. "
        f"Churn rate: {targets_df['is_churned'].mean():.2%}, "
        f"Avg future CLV: £{targets_df['future_clv'].mean():.2f}"
    )
    return targets_df


def compute_rfm_and_spend_features(
    df_obs: pd.DataFrame, cutoff_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Computes RFM (Recency, Frequency, Monetary) and core spending metrics strictly
    before the cutoff date.
    """
    # Filter positive purchase rows for invoice metrics
    pos_df = df_obs[(df_obs["quantity"] > 0) & (~df_obs["is_cancellation"])]
    
    # 1. Group aggregations on positive purchases
    pos_agg = pos_df.groupby("customer_id").agg(
        frequency=("invoice", "nunique"),
        last_purchase_date=("invoice_date", "max"),
        first_purchase_date=("invoice_date", "min"),
        total_items_bought=("quantity", "sum"),
        unique_skus_bought=("stock_code", "nunique"),
    )
    
    # 2. Net monetary spend from ALL observation records (including returns)
    net_spend = df_obs.groupby("customer_id")["line_total"].sum().rename("monetary_total")
    
    # Merge aggregations
    features = pos_agg.merge(net_spend, on="customer_id", how="left")
    
    # Recency and Customer Age in days
    features["recency"] = (cutoff_date - features["last_purchase_date"]).dt.total_seconds() / 86400.0
    features["customer_age_days"] = (cutoff_date - features["first_purchase_date"]).dt.total_seconds() / 86400.0
    
    # Averages
    features["monetary_avg"] = features["monetary_total"] / features["frequency"].clip(lower=1)
    features["avg_basket_size"] = features["total_items_bought"] / features["frequency"].clip(lower=1)
    features["historical_clv"] = features["monetary_total"].clip(lower=0.0)
    
    features.drop(columns=["last_purchase_date", "first_purchase_date"], inplace=True)
    return features.reset_index()


def compute_behavioral_dynamics(df_obs: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """
    Computes purchase timing variance (std of inter-purchase gaps) and purchase acceleration/trend.
    """
    pos_df = df_obs[(df_obs["quantity"] > 0) & (~df_obs["is_cancellation"])].copy()
    
    # Compute distinct purchase dates per customer
    distinct_orders = (
        pos_df.groupby(["customer_id", "invoice"])["invoice_date"]
        .min()
        .reset_index()
        .sort_values(by=["customer_id", "invoice_date"])
    )
    
    # Inter-purchase interval calculations
    distinct_orders["prev_date"] = distinct_orders.groupby("customer_id")["invoice_date"].shift(1)
    distinct_orders["gap_days"] = (
        distinct_orders["invoice_date"] - distinct_orders["prev_date"]
    ).dt.total_seconds() / 86400.0
    
    timing_stats = distinct_orders.groupby("customer_id")["gap_days"].agg(
        interpurchase_interval_mean="mean",
        interpurchase_interval_std="std",
    ).fillna(0.0).reset_index()
    
    # Purchase trend: compare orders in recent 90d before cutoff vs previous 90d (90-180d)
    date_90d_ago = cutoff_date - pd.Timedelta(days=90)
    date_180d_ago = cutoff_date - pd.Timedelta(days=180)
    
    orders_last_90d = (
        pos_df[pos_df["invoice_date"] >= date_90d_ago]
        .groupby("customer_id")["invoice"]
        .nunique()
        .rename("orders_recent_90d")
    )
    orders_prior_90d = (
        pos_df[(pos_df["invoice_date"] >= date_180d_ago) & (pos_df["invoice_date"] < date_90d_ago)]
        .groupby("customer_id")["invoice"]
        .nunique()
        .rename("orders_prior_90d")
    )
    
    trend_df = pd.DataFrame({"customer_id": sorted(df_obs["customer_id"].dropna().unique())})
    trend_df = trend_df.merge(orders_last_90d, on="customer_id", how="left").fillna(0)
    trend_df = trend_df.merge(orders_prior_90d, on="customer_id", how="left").fillna(0)
    
    # Trend slope: (recent - prior) / (recent + prior + 1) normalized between -1 and 1
    trend_df["purchase_frequency_trend"] = (
        (trend_df["orders_recent_90d"] - trend_df["orders_prior_90d"])
        / (trend_df["orders_recent_90d"] + trend_df["orders_prior_90d"] + 1.0)
    )
    trend_df.drop(columns=["orders_recent_90d", "orders_prior_90d"], inplace=True)
    
    dynamics = timing_stats.merge(trend_df, on="customer_id", how="right").fillna(0.0)
    return dynamics


def compute_risk_and_preference_features(df_obs: pd.DataFrame) -> pd.DataFrame:
    """
    Computes return rate, cancellation rate, seasonal Q4 concentration, and discount/price sensitivity.
    """
    cust_ids = sorted(df_obs["customer_id"].dropna().unique())
    base_df = pd.DataFrame({"customer_id": cust_ids})
    
    # Return rate: returned items / total purchased items
    return_items = (
        df_obs[df_obs["quantity"] < 0]
        .groupby("customer_id")["quantity"]
        .sum()
        .abs()
        .rename("returned_qty")
    )
    bought_items = (
        df_obs[df_obs["quantity"] > 0]
        .groupby("customer_id")["quantity"]
        .sum()
        .rename("bought_qty")
    )
    
    # Cancellations: invoices flagged as cancellation / total unique invoices
    total_invs = df_obs.groupby("customer_id")["invoice"].nunique().rename("total_invoices")
    cancel_invs = (
        df_obs[df_obs["is_cancellation"]]
        .groupby("customer_id")["invoice"]
        .nunique()
        .rename("cancel_invoices")
    )
    
    # Seasonal concentration: proportion of spend occurring in Q4 (months 10, 11, 12)
    q4_spend = (
        df_obs[df_obs["invoice_date"].dt.month.isin([10, 11, 12])]
        .groupby("customer_id")["line_total"]
        .sum()
        .rename("q4_spend")
    )
    total_spend = df_obs.groupby("customer_id")["line_total"].sum().rename("total_spend")
    
    # Discount sensitivity proxy: share of items bought where unit price is below median product price (£2.10)
    median_price = df_obs[df_obs["quantity"] > 0]["price"].median()
    low_price_items = (
        df_obs[(df_obs["quantity"] > 0) & (df_obs["price"] < median_price)]
        .groupby("customer_id")["quantity"]
        .sum()
        .rename("discounted_qty")
    )
    
    # Country / Domestic feature
    primary_country = (
        df_obs.groupby("customer_id")["country"]
        .agg(lambda x: x.mode()[0] if not x.empty else "United Kingdom")
        .rename("primary_country")
    )
    
    # Merge all
    df_feats = (
        base_df.merge(return_items, on="customer_id", how="left").fillna(0)
        .merge(bought_items, on="customer_id", how="left").fillna(0)
        .merge(total_invs, on="customer_id", how="left").fillna(1)
        .merge(cancel_invs, on="customer_id", how="left").fillna(0)
        .merge(q4_spend, on="customer_id", how="left").fillna(0)
        .merge(total_spend, on="customer_id", how="left").fillna(0)
        .merge(low_price_items, on="customer_id", how="left").fillna(0)
        .merge(primary_country, on="customer_id", how="left")
    )
    
    df_feats["return_rate"] = df_feats["returned_qty"] / (df_feats["bought_qty"] + df_feats["returned_qty"] + 1e-5)
    df_feats["cancellation_rate"] = df_feats["cancel_invoices"] / df_feats["total_invoices"].clip(lower=1)
    df_feats["seasonal_concentration"] = (df_feats["q4_spend"] / df_feats["total_spend"].clip(lower=1.0)).clip(0.0, 1.0)
    df_feats["discount_sensitivity"] = (df_feats["discounted_qty"] / (df_feats["bought_qty"] + 1e-5)).clip(0.0, 1.0)
    df_feats["is_uk_customer"] = (df_feats["primary_country"] == "United Kingdom").astype(int)
    
    cols_to_drop = [
        "returned_qty", "bought_qty", "total_invoices", "cancel_invoices",
        "q4_spend", "total_spend", "discounted_qty", "primary_country"
    ]
    df_feats.drop(columns=cols_to_drop, inplace=True)
    return df_feats


def compute_engagement_score(df_features: pd.DataFrame) -> pd.Series:
    """
    Computes a composite, interpretable engagement score between 0 and 100
    using percentile ranks of Recency (inverted), Frequency, and Monetary Total.
    """
    config = load_config()
    weights = config.get("features", {}).get("weights", {})
    r_w = weights.get("recency_weight", 0.4)
    f_w = weights.get("frequency_weight", 0.3)
    m_w = weights.get("monetary_weight", 0.3)

    # Invert recency so lower recency (more recent) gets higher score
    r_rank = 1.0 - df_features["recency"].rank(pct=True)
    f_rank = df_features["frequency"].rank(pct=True)
    m_rank = df_features["monetary_total"].rank(pct=True)

    score = 100.0 * (r_w * r_rank + f_w * f_rank + m_w * m_rank)
    return score.round(2)


def build_customer_feature_table(
    df_cleaned: pd.DataFrame,
    cutoff_date_str: Optional[str] = None,
    save_processed: bool = True,
) -> pd.DataFrame:
    """
    Master pipeline that takes cleaned transaction logs, strictly splits them at cutoff_date,
    computes all customer features on the observation window, computes targets on the prediction window,
    and returns a clean, leak-free customer-level dataset.
    """
    config = load_config()
    if cutoff_date_str is None:
        cutoff_date_str = config.get("features", {}).get("temporal", {}).get("cutoff_date", "2011-09-01")
        
    cutoff_date = pd.Timestamp(cutoff_date_str)
    logger.info(f"Building customer feature table with Point-in-Time cutoff: {cutoff_date}")

    df_obs, df_pred = extract_temporal_windows(df_cleaned, cutoff_date)

    # Compute feature groups strictly on df_obs
    logger.info("Computing RFM and spend features...")
    rfm_df = compute_rfm_and_spend_features(df_obs, cutoff_date)
    
    logger.info("Computing behavioral dynamics (gap std, trend slope)...")
    dynamics_df = compute_behavioral_dynamics(df_obs, cutoff_date)
    
    logger.info("Computing risk & preference metrics (returns, discounts, seasonal)...")
    risk_df = compute_risk_and_preference_features(df_obs)
    
    # Merge feature groups
    customer_df = rfm_df.merge(dynamics_df, on="customer_id", how="inner")
    customer_df = customer_df.merge(risk_df, on="customer_id", how="inner")
    
    # Add composite engagement score
    customer_df["engagement_score"] = compute_engagement_score(customer_df)
    
    # Compute targets strictly on df_pred
    logger.info("Computing churn & future CLV targets...")
    targets_df = compute_targets(df_obs, df_pred)
    
    final_df = customer_df.merge(targets_df, on="customer_id", how="inner")
    
    # Clean inf/nan if any
    final_df.fillna(0.0, inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    
    logger.info(f"Feature table constructed: {len(final_df):,} customers, {final_df.shape[1]} columns.")
    
    if save_processed:
        root = get_project_root()
        proc_dir = root / config.get("paths", {}).get("processed_dir", "data/processed")
        proc_dir.mkdir(parents=True, exist_ok=True)
        out_path = proc_dir / "customer_features.parquet"
        logger.info(f"Saving processed customer features to: {out_path}")
        final_df.to_parquet(out_path, index=False)

    return final_df


if __name__ == "__main__":
    from src.data.loader import load_raw_data
    from src.data.cleaner import run_cleaning_pipeline

    root = get_project_root()
    interim_path = root / "data/interim/cleaned_transactions.parquet"
    
    if interim_path.exists():
        logger.info("Loading cleaned transactions from interim cache...")
        df_cleaned = pd.read_parquet(interim_path)
    else:
        logger.info("Cleaned transactions cache not found. Running loader and cleaner...")
        df_raw = load_raw_data()
        df_cleaned = run_cleaning_pipeline(df_raw)

    features_df = build_customer_feature_table(df_cleaned)
    print("\n--- Processed Feature Table Summary ---")
    print(f"Total Customers: {len(features_df):,}")
    print(f"Feature Columns ({features_df.shape[1]}): {list(features_df.columns)}")
    print(f"Churn Rate: {features_df['is_churned'].mean():.2%}")
    print(f"Mean Future CLV: £{features_df['future_clv'].mean():.2f}")
    print("\nSample Preview:")
    print(features_df[["customer_id", "recency", "frequency", "monetary_total", "engagement_score", "is_churned", "future_clv"]].head())
