import pytest
import pandas as pd

from src.features.builder import (
    extract_temporal_windows,
    compute_targets,
    compute_rfm_and_spend_features,
    compute_engagement_score,
)


@pytest.fixture
def mock_transactions():
    cutoff = pd.Timestamp("2011-09-01")
    data = [
        # Customer 1: active before cutoff, also active after cutoff (retained, is_churned=0)
        {"invoice": "1001", "stock_code": "A", "description": "ITEM A", "quantity": 2, "price": 10.0, "customer_id": 1.0, "invoice_date": pd.Timestamp("2011-06-01"), "line_total": 20.0, "is_cancellation": False, "country": "United Kingdom"},
        {"invoice": "1002", "stock_code": "B", "description": "ITEM B", "quantity": 1, "price": 15.0, "customer_id": 1.0, "invoice_date": pd.Timestamp("2011-08-15"), "line_total": 15.0, "is_cancellation": False, "country": "United Kingdom"},
        {"invoice": "1003", "stock_code": "C", "description": "ITEM C", "quantity": 3, "price": 10.0, "customer_id": 1.0, "invoice_date": pd.Timestamp("2011-10-01"), "line_total": 30.0, "is_cancellation": False, "country": "United Kingdom"},
        
        # Customer 2: active before cutoff, inactive after cutoff (churned, is_churned=1)
        {"invoice": "2001", "stock_code": "A", "description": "ITEM A", "quantity": 5, "price": 20.0, "customer_id": 2.0, "invoice_date": pd.Timestamp("2011-01-10"), "line_total": 100.0, "is_cancellation": False, "country": "Germany"},
        
        # Customer 3: joins strictly after cutoff (new customer, should NOT be evaluated at cutoff)
        {"invoice": "3001", "stock_code": "D", "description": "ITEM D", "quantity": 1, "price": 50.0, "customer_id": 3.0, "invoice_date": pd.Timestamp("2011-11-01"), "line_total": 50.0, "is_cancellation": False, "country": "France"},
    ]
    return pd.DataFrame(data), cutoff


def test_temporal_window_split_no_leakage(mock_transactions):
    df, cutoff = mock_transactions
    df_obs, df_pred = extract_temporal_windows(df, cutoff)

    # Verification: Observation window strictly before cutoff
    assert (df_obs["invoice_date"] < cutoff).all(), "Leakage detected: obs records on or after cutoff!"
    # Verification: Prediction window strictly on or after cutoff
    assert (df_pred["invoice_date"] >= cutoff).all(), "Leakage detected: pred records before cutoff!"


def test_leak_free_target_calculation(mock_transactions):
    df, cutoff = mock_transactions
    df_obs, df_pred = extract_temporal_windows(df, cutoff)
    targets = compute_targets(df_obs, df_pred)

    # Customer 3 should not be in target evaluation cohort as they weren't historical customers
    assert set(targets["customer_id"]) == {1.0, 2.0}

    # Customer 1 purchased after cutoff -> not churned
    c1 = targets[targets["customer_id"] == 1.0].iloc[0]
    assert c1["is_churned"] == 0
    assert c1["future_clv"] == 30.0

    # Customer 2 did not purchase after cutoff -> churned
    c2 = targets[targets["customer_id"] == 2.0].iloc[0]
    assert c2["is_churned"] == 1
    assert c2["future_clv"] == 0.0


def test_rfm_calculation(mock_transactions):
    df, cutoff = mock_transactions
    df_obs, _ = extract_temporal_windows(df, cutoff)
    rfm = compute_rfm_and_spend_features(df_obs, cutoff)

    # Customer 1: invoices 1001 & 1002 (frequency=2, monetary=35.0)
    c1 = rfm[rfm["customer_id"] == 1.0].iloc[0]
    assert c1["frequency"] == 2
    assert c1["monetary_total"] == 35.0
    assert c1["recency"] == pytest.approx((cutoff - pd.Timestamp("2011-08-15")).total_seconds() / 86400.0)


def test_engagement_score_bounds():
    df = pd.DataFrame({
        "recency": [10.0, 50.0, 200.0],
        "frequency": [15, 5, 1],
        "monetary_total": [5000.0, 500.0, 50.0],
    })
    scores = compute_engagement_score(df)
    assert (scores >= 0).all()
    assert (scores <= 100).all()
    # Most active customer should have highest engagement
    assert scores.iloc[0] > scores.iloc[2]
