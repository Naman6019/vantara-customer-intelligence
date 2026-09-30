from pathlib import Path
from typing import List, Dict, Any
import io
import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File
import shap

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger
from src.models.data_split import FEATURE_COLS
from api.schemas.customer import (
    CustomerFeaturesInput,
    PredictionResponse,
    DriverImpact,
    ModelMetadataResponse,
)

logger = get_logger("api_router")
router = APIRouter()

# Load artifacts once at module startup
root = get_project_root()
art_dir = root / "models_artifacts"

# Churn model
churn_model_path = art_dir / "churn_xgboost.joblib"
if not churn_model_path.exists():
    churn_model_path = art_dir / "best_churn_model.joblib"
churn_model = joblib.load(churn_model_path)
shap_explainer = shap.TreeExplainer(churn_model)

# CLV model
clv_model = joblib.load(art_dir / "best_clv_model.joblib")
scaler = joblib.load(art_dir / "scaler.joblib")

# Segmentation model
kmeans_model = joblib.load(art_dir / "kmeans_model.joblib")
seg_scaler = joblib.load(art_dir / "segmentation_scaler.joblib")
personas = joblib.load(art_dir / "segment_personas.joblib")


@router.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "Vantara Customer Behavior Prediction Platform"}


@router.get("/metadata", response_model=ModelMetadataResponse)
def get_model_metadata() -> ModelMetadataResponse:
    """Returns model versioning, architecture, and validation metrics."""
    return ModelMetadataResponse(
        platform="Vantara Retail Solutions Customer Behavior Prediction Platform",
        version="1.0.0",
        churn_model=type(churn_model).__name__,
        clv_model=type(clv_model).__name__,
        segmentation_model="K-Means (k=4)",
        test_roc_auc=0.7846,
        test_clv_r2=0.8447,
        features_expected=FEATURE_COLS,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict_single_customer(features: CustomerFeaturesInput) -> PredictionResponse:
    """
    Real-time single-customer inference returning churn probability, future CLV,
    customer persona segment, and top SHAP risk drivers with plain-English narrative.
    """
    try:
        # Convert input to DataFrame
        row_dict = features.model_dump()
        cust_id = row_dict.pop("customer_id", None)
        df_row = pd.DataFrame([row_dict])[FEATURE_COLS]

        # Churn prediction
        prob = float(churn_model.predict_proba(df_row)[0, 1])
        is_risk = bool(prob >= 0.50)
        risk_tier = "High" if prob >= 0.65 else ("Moderate" if prob >= 0.40 else "Low")

        # CLV prediction
        # Check if model used scaled or unscaled features (e.g. Ridge uses scaled)
        if type(clv_model).__name__ == "Ridge":
            scaled_row = pd.DataFrame(scaler.transform(df_row), columns=FEATURE_COLS)
            future_clv = float(np.maximum(0, clv_model.predict(scaled_row)[0]))
        else:
            future_clv = float(np.maximum(0, clv_model.predict(df_row)[0]))

        # Segmentation prediction
        seg_features = ["recency", "frequency", "monetary_total", "return_rate", "engagement_score"]
        seg_scaled = seg_scaler.transform(df_row[seg_features])
        cluster = int(kmeans_model.predict(seg_scaled)[0])
        persona = personas.get(cluster, f"Segment {cluster}")

        # Local SHAP explanation
        shap_res = shap_explainer(df_row)
        shap_vals = shap_res.values[0]

        impacts = []
        for col, val, s_val in zip(FEATURE_COLS, df_row.iloc[0], shap_vals):
            impacts.append(
                DriverImpact(
                    feature=col,
                    value=float(val),
                    shap_impact=round(float(s_val), 4),
                    direction="Increases Churn" if s_val > 0 else "Decreases Churn",
                )
            )
        impacts.sort(key=lambda x: abs(x.shap_impact), reverse=True)
        top_drivers = impacts[:4]

        # Generate plain English narrative
        pos_drivers = [d for d in top_drivers if d.shap_impact > 0]
        neg_drivers = [d for d in top_drivers if d.shap_impact < 0]
        narrative = f"Customer flagged as {risk_tier} Churn Risk ({prob:.1%} probability). "
        if pos_drivers:
            narrative += f"Primary warning signs: {pos_drivers[0].feature.replace('_', ' ').title()} ({pos_drivers[0].value:.1f}, +{pos_drivers[0].shap_impact:.2f}). "
        if neg_drivers:
            narrative += f"Mitigating factor: {neg_drivers[0].feature.replace('_', ' ').title()} ({neg_drivers[0].value:.1f}, {neg_drivers[0].shap_impact:.2f})."

        return PredictionResponse(
            customer_id=cust_id,
            churn_probability=round(prob, 4),
            is_churn_risk=is_risk,
            risk_tier=risk_tier,
            predicted_future_clv=round(future_clv, 2),
            predicted_segment=persona,
            narrative_explanation=narrative.strip(),
            top_risk_drivers=top_drivers,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-score")
async def batch_score_csv(file: UploadFile = File(...)) -> List[Dict[str, Any]]:
    """
    Accepts an uploaded CSV file containing customer feature records,
    computes churn risk, future CLV, and segmentation for all rows.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        missing = [col for col in FEATURE_COLS if col not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded CSV is missing required feature columns: {missing}"
            )

        X = df[FEATURE_COLS]
        probs = churn_model.predict_proba(X)[:, 1]

        if type(clv_model).__name__ == "Ridge":
            scaled_X = scaler.transform(X)
            clvs = np.maximum(0, clv_model.predict(scaled_X))
        else:
            clvs = np.maximum(0, clv_model.predict(X))

        seg_features = ["recency", "frequency", "monetary_total", "return_rate", "engagement_score"]
        seg_scaled = seg_scaler.transform(X[seg_features])
        clusters = kmeans_model.predict(seg_scaled)
        assigned_personas = [personas.get(c, f"Segment {c}") for c in clusters]

        results = []
        cust_ids = df["customer_id"].tolist() if "customer_id" in df.columns else range(len(df))
        
        for cid, prob, clv_val, pers in zip(cust_ids, probs, clvs, assigned_personas):
            results.append({
                "customer_id": cid,
                "churn_probability": round(float(prob), 4),
                "is_churn_risk": bool(prob >= 0.50),
                "predicted_future_clv": round(float(clv_val), 2),
                "predicted_segment": pers,
            })

        return results
    except Exception as e:
        logger.error(f"Batch scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
