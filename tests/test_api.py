from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metadata_endpoint():
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "churn_model" in data
    assert "clv_model" in data
    assert "segmentation_model" in data
    assert data["test_roc_auc"] >= 0.70
    assert data["test_clv_r2"] >= 0.60


def test_predict_single_customer_valid():
    payload = {
        "customer_id": 12345.0,
        "frequency": 8,
        "total_items_bought": 250,
        "unique_skus_bought": 45,
        "monetary_total": 1250.50,
        "recency": 14.5,
        "customer_age_days": 320.0,
        "monetary_avg": 156.31,
        "avg_basket_size": 31.25,
        "historical_clv": 1250.50,
        "interpurchase_interval_mean": 28.5,
        "interpurchase_interval_std": 12.1,
        "purchase_frequency_trend": 0.25,
        "return_rate": 0.02,
        "cancellation_rate": 0.0,
        "seasonal_concentration": 0.35,
        "discount_sensitivity": 0.15,
        "is_uk_customer": 1,
        "engagement_score": 82.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert "predicted_future_clv" in data
    assert data["predicted_future_clv"] >= 0
    assert "predicted_segment" in data
    assert "narrative_explanation" in data
    assert len(data["top_risk_drivers"]) > 0


def test_predict_validation_error():
    # Negative frequency or missing required fields should trigger validation error
    invalid_payload = {
        "frequency": -5,
        "monetary_total": 100.0,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity
