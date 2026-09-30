"""Monitoring and revenue endpoints (plan D6/D7)."""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

from src.serving import db

router = APIRouter()


@router.get("/monitoring/drift")
def monitoring_drift() -> List[Dict[str, Any]]:
    """Per-feature PSI from the most recent drift report."""
    from sqlalchemy import text

    if db.get_engine() is None:
        db.ensure_engine()
    engine = db.get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    with engine.connect() as conn:
        latest = conn.execute(text("SELECT MAX(batch_date) AS d FROM drift_reports")).fetchone()
        batch_date = latest._mapping["d"] if latest else None
        if batch_date is None:
            return []
        rows = conn.execute(
            text(
                "SELECT feature, psi, drifted, batch_date FROM drift_reports "
                "WHERE batch_date = :d ORDER BY psi DESC"
            ),
            {"d": batch_date},
        ).fetchall()
    return [dict(r._mapping) for r in rows]


@router.get("/revenue/monthly")
def revenue_monthly(request: Request) -> List[Dict[str, Any]]:
    """Monthly revenue read from the cleaned transactions when available."""
    import pandas as pd

    from src.utils.config import get_project_root

    path = get_project_root() / "data/interim/cleaned_transactions.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Cleaned transactions not available")
    df = pd.read_parquet(path, columns=["invoice_date", "line_total"])
    monthly = (
        df.groupby(df["invoice_date"].dt.to_period("M"))["line_total"]
        .sum()
        .reset_index()
        .rename(columns={"invoice_date": "month", "line_total": "revenue"})
    )
    monthly["month"] = monthly["month"].astype(str)
    return monthly.to_dict(orient="records")


@router.get("/monitoring/performance")
def monitoring_performance(request: Request) -> Dict[str, Any]:
    """Delayed-label performance recorded on the model card (AUC/recall per split)."""
    card = request.app.state.artifacts["card"]
    return {
        "model_version": card["model_version"],
        "validation": card.get("validation_metrics", {}),
        "test": card.get("test_metrics", {}),
        "clv_validation": card.get("clv_validation_metrics", {}),
        "clv_test": card.get("clv_test_metrics", {}),
    }
