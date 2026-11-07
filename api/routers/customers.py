"""Customer, segment and value-at-risk endpoints backed by PostgreSQL (plan D6)."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from api.state import get_artifacts
from src.serving import db
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger("api_customers")


_LIME_BACKGROUND: Optional[Any] = None


def _lime_background(engine, feature_cols: List[str], n: int = 500):
    """Stored feature vectors of up to ``n`` scored customers, loaded once per process."""
    global _LIME_BACKGROUND
    if _LIME_BACKGROUND is None:
        import pandas as pd
        from sqlalchemy import text

        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT features FROM predictions WHERE features IS NOT NULL ORDER BY customer_id LIMIT :n"),
                {"n": n},
            ).fetchall()
        vectors = [_as_dict(r._mapping["features"]) for r in rows]
        _LIME_BACKGROUND = pd.DataFrame(
            [{c: float(v.get(c, 0.0) or 0.0) for c in feature_cols} for v in vectors]
        )
    return _LIME_BACKGROUND


def _rows_to_dicts(rows) -> List[Dict[str, Any]]:
    return [dict(r._mapping) for r in rows]


def _as_dict(value: Any) -> Dict[str, Any]:
    """JSON columns come back as dicts (PostgreSQL) or strings (SQLite)."""
    import json

    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _filters(
    segment: Optional[str], country: Optional[str], value_tier: Optional[str]
) -> tuple[str, Dict[str, Any]]:
    """SQL conditions (on predictions `p` and segments `s`) shared by the list views."""
    sql, params = "", {}
    if segment:
        sql += " AND s.persona LIKE :segment"
        params["segment"] = f"%{segment}%"
    if country:
        sql += " AND p.customer_id IN (SELECT customer_id FROM customers WHERE primary_country = :country)"
        params["country"] = country
    if value_tier:
        sql += " AND p.value_tier = :value_tier"
        params["value_tier"] = value_tier.lower()
    return sql, params


@router.get("/customers/at-risk")
def at_risk_customers(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    segment: Optional[str] = None,
    country: Optional[str] = None,
    value_tier: Optional[str] = Query(None, pattern="^(?i)(low|mid|high)$"),
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
    where, params = _filters(segment, country, value_tier)
    sql += where + " ORDER BY p.value_at_risk DESC LIMIT :limit"
    params["limit"] = limit

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


@router.get("/customers/{customer_id}/recommendations")
def customer_recommendations(request: Request, customer_id: float) -> List[Dict[str, Any]]:
    """Recommended next products: personalised, or labelled best-sellers if there is no history."""
    from sqlalchemy import text

    from src.recommendations.products import recommend_for_customer

    trained = recommend_for_customer(customer_id, top_k=10)
    if trained:
        return trained

    if db.get_engine() is None:
        db.ensure_engine()
    engine = db.get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    # Products this customer already bought.
    with engine.connect() as conn:
        bought = conn.execute(
            text("SELECT DISTINCT product_id FROM customer_products WHERE customer_id = :cid"),
            {"cid": customer_id},
        ).fetchall()
        bought_ids = {r[0] for r in bought}
        top = conn.execute(
            text(
                "SELECT product_id, COUNT(*) AS buyers, SUM(quantity) AS units "
                "FROM customer_products WHERE product_id != '' "
                "GROUP BY product_id ORDER BY buyers DESC LIMIT 40"
            )
        ).fetchall()

    return [
        {"product_id": r[0], "buyers": int(r[1]), "units": float(r[2] or 0), "source": "popularity"}
        for r in top
        if r[0] not in bought_ids
    ][:10]


@router.get("/customers/{customer_id}/explain")
def explain_customer(request: Request, customer_id: float) -> Dict[str, Any]:
    """Per-customer SHAP and LIME explanation, plus a plain-language narrative.

    Both explainers use the served model and the stored feature vector, so the
    dashboard never loads model artifacts directly (PRD acceptance criterion 4).
    """
    from sqlalchemy import text

    if db.get_engine() is None:
        db.ensure_engine()
    engine = db.get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT features, calibrated_probability, risk_tier "
                "FROM predictions WHERE customer_id = :cid ORDER BY scored_at DESC LIMIT 1"
            ),
            {"cid": customer_id},
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    import pandas as pd

    from src.explainability.local import lime_drivers as lime_for
    from src.explainability.local import shap_drivers as shap_for

    art = get_artifacts(request)
    feature_cols = art["card"]["feature_cols"]
    vector = {c: float(_as_dict(row._mapping["features"]).get(c, 0.0) or 0.0) for c in feature_cols}
    X = pd.DataFrame([vector])[feature_cols]

    shap_drivers = shap_for(art, X)[0][:8]

    # LIME on the same customer, sampling around the scored customer base.
    lime_drivers: List[Dict[str, Any]] = []
    try:
        lime_drivers = lime_for(art, X, _lime_background(engine, feature_cols))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"LIME explanation skipped: {exc}")

    prob = float(row._mapping["calibrated_probability"])
    pos = [d for d in shap_drivers if d["shap_impact"] > 0]
    narrative = f"Calibrated churn risk {prob:.1%} ({row._mapping['risk_tier']}). "
    if pos:
        narrative += f"Top warning sign: {pos[0]['feature'].replace('_', ' ')}."

    # Beyond PRD P1: probabilistic P(alive) from BG/NBD for the same customer.
    p_alive = None
    try:
        import json as _json

        import pandas as _pd

        from src.models.probabilistic_clv import (
            build_rfm_summary,
            deserialize_models,
            predict,
        )
        from src.utils.config import get_project_root, load_config

        path = get_project_root() / "models_artifacts" / "probabilistic_clv.json"
        cleaned = get_project_root() / "data/interim/cleaned_transactions.parquet"
        if path.exists() and cleaned.exists():
            df_clean = _pd.read_parquet(cleaned)
            cutoff = _pd.Timestamp(
                load_config().get("features", {}).get("temporal", {}).get("cutoff_date", "2011-09-01")
            )
            summary = build_rfm_summary(df_clean, cutoff)
            row_summary = summary[summary["customer_id"] == customer_id]
            if not row_summary.empty:
                models = deserialize_models(_json.loads(path.read_text(encoding="utf-8")))
                p_alive = float(predict(models, row_summary)["p_alive"].iloc[0])
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"P(alive) skipped: {exc}")

    return {
        "customer_id": customer_id,
        "risk_tier": row._mapping["risk_tier"],
        "calibrated_probability": prob,
        "p_alive": round(p_alive, 4) if p_alive is not None else None,
        "shap": shap_drivers,
        "lime": lime_drivers,
        "narrative": narrative.strip(),
    }


@router.get("/segments")
def segments_overview(
    request: Request,
    country: Optional[str] = None,
    value_tier: Optional[str] = Query(None, pattern="^(?i)(low|mid|high)$"),
) -> List[Dict[str, Any]]:
    """Per-persona summary statistics, optionally for one country and value tier."""
    from sqlalchemy import text

    if db.get_engine() is None:
        db.ensure_engine()
    engine = db.get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    where, params = _filters(None, country, value_tier)
    sql = f"""
        SELECT s.persona,
               COUNT(*) AS n_customers,
               SUM(CASE WHEN p.risk_tier = 'High' THEN 1 ELSE 0 END) AS n_high_risk,
               AVG(p.calibrated_probability) AS avg_churn_prob,
               AVG(p.predicted_value) AS avg_predicted_value,
               SUM(p.value_at_risk) AS total_value_at_risk
        FROM segments s
        LEFT JOIN predictions p ON p.customer_id = s.customer_id
        WHERE 1=1 {where}
        GROUP BY s.persona
        ORDER BY total_value_at_risk DESC
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return _rows_to_dicts(rows)
