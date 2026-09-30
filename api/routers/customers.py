"""Customer, segment and value-at-risk endpoints backed by PostgreSQL (plan D6)."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from api.state import get_artifacts
from src.serving import db

router = APIRouter()


def _rows_to_dicts(rows) -> List[Dict[str, Any]]:
    return [dict(r._mapping) for r in rows]


@router.get("/customers/at-risk")
def at_risk_customers(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    segment: Optional[str] = None,
    country: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Value-at-risk leaderboard, sorted by calibrated P(churn) x predicted value."""
    from sqlalchemy import text

    if db.get_engine() is None:
        db.ensure_engine()
    engine = db.get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    sql = """
        SELECT p.customer_id, p.calibrated_probability, p.predicted_value, p.value_at_risk,
               p.risk_tier, p.value_tier, s.persona, r.action, r.channel, r.priority_score
        FROM predictions p
        LEFT JOIN segments s ON s.customer_id = p.customer_id
        LEFT JOIN recommendations r ON r.customer_id = p.customer_id
        WHERE 1=1
    """
    params: Dict[str, Any] = {"limit": limit}
    if segment:
        sql += " AND s.persona LIKE :segment"
        params["segment"] = f"%{segment}%"
    if country:
        sql += " AND p.customer_id IN (SELECT customer_id FROM customers WHERE primary_country = :country)"
        params["country"] = country
    sql += " ORDER BY p.value_at_risk DESC LIMIT :limit"

    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return _rows_to_dicts(rows)


@router.get("/customers/{customer_id}")
def get_customer(request: Request, customer_id: float) -> Dict[str, Any]:
    """Full Customer-360 record: prediction, segment, recommendation and SHAP drivers."""
    from sqlalchemy import text

    if db.get_engine() is None:
        db.ensure_engine()
    engine = db.get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    sql = """
        SELECT p.customer_id, p.calibrated_probability, p.churn_probability, p.predicted_value,
               p.value_at_risk, p.risk_tier, p.value_tier, p.shap_top,
               s.persona, s.kmeans_cluster, r.action, r.channel, r.reasons, r.priority_score
        FROM predictions p
        LEFT JOIN segments s ON s.customer_id = p.customer_id
        LEFT JOIN recommendations r ON r.customer_id = p.customer_id
        WHERE p.customer_id = :cid
        ORDER BY p.scored_at DESC
        LIMIT 1
    """
    with engine.connect() as conn:
        row = conn.execute(text(sql), {"cid": customer_id}).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return dict(row._mapping)


@router.get("/segments")
def segments_overview(request: Request) -> List[Dict[str, Any]]:
    """Aggregate the segmentation table into per-persona summary statistics."""
    from sqlalchemy import text

    if db.get_engine() is None:
        db.ensure_engine()
    engine = db.get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    sql = """
        SELECT s.persona,
               COUNT(*) AS n_customers,
               AVG(p.calibrated_probability) AS avg_churn_prob,
               AVG(p.predicted_value) AS avg_predicted_value,
               SUM(p.value_at_risk) AS total_value_at_risk
        FROM segments s
        LEFT JOIN predictions p ON p.customer_id = s.customer_id
        GROUP BY s.persona
        ORDER BY total_value_at_risk DESC
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()
    return _rows_to_dicts(rows)
