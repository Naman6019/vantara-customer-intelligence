"""Regenerate docs/architecture_diagram.png, docs/er_diagram.png and docs/workflow_diagram.png.

Pure matplotlib so the diagrams can be rebuilt anywhere (no Graphviz needed).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

NAVY, INDIGO, EMERALD, AMBER, SLATE = "#0F172A", "#4F46E5", "#059669", "#D97706", "#475569"


def _box(ax, x, y, w, h, text, color, text_color="white", size=9):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1, edgecolor=color, facecolor=color,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=text_color, fontsize=size, wrap=True)


def _arrow(ax, x1, y1, x2, y2, color=SLATE):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))


def make_architecture():
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.text(0.2, 7.0, "Vantara — Customer Behavior Prediction Platform (architecture)",
            fontsize=13, fontweight="bold", color=NAVY)

    _box(ax, 0.3, 5.6, 2.2, 0.8, "Raw UCI Online Retail II\n(1.07M rows)", NAVY)
    _box(ax, 2.9, 5.6, 2.4, 0.8, "Clean + validate\n(flag returns, pandera)", SLATE)
    _box(ax, 5.7, 5.6, 2.6, 0.8, "Leak-free features\n+ stacked snapshots", INDIGO)
    _box(ax, 8.7, 5.6, 2.2, 0.8, "Train / select\n(Optuna, fwd-chain CV)", INDIGO)
    _box(ax, 11.3, 5.6, 1.4, 0.8, "MLflow\n+ model card", EMERALD, size=8)

    _arrow(ax, 2.5, 6.0, 2.9, 6.0)
    _arrow(ax, 5.3, 6.0, 5.7, 6.0)
    _arrow(ax, 8.3, 6.0, 8.7, 6.0)
    _arrow(ax, 10.9, 6.0, 11.3, 6.0)

    _box(ax, 2.9, 4.1, 2.4, 0.8, "Calibrate (isotonic)\n+ threshold 0.70 recall", AMBER, size=8)
    _box(ax, 5.7, 4.1, 2.6, 0.8, "Batch scorer\n(value at risk + rules)", INDIGO)
    _box(ax, 8.7, 4.1, 2.2, 0.8, "Drift (PSI)\n+ prediction log", SLATE)
    _arrow(ax, 10.0, 5.6, 10.0, 4.9)
    _arrow(ax, 6.8, 5.6, 6.8, 4.9)
    _arrow(ax, 7.0, 4.1, 8.7, 4.5)

    _box(ax, 5.7, 2.5, 2.6, 0.9, "PostgreSQL\npredictions · segments ·\nrecommendations · drift", NAVY)
    _arrow(ax, 7.0, 4.1, 7.0, 3.4)

    _box(ax, 3.4, 1.1, 2.4, 0.9, "FastAPI\n(/customers, /predict,\n/campaign/plan, /metrics)", EMERALD)
    _box(ax, 7.4, 1.1, 2.4, 0.9, "Streamlit\ndashboard (API-only)", EMERALD)
    _arrow(ax, 6.8, 2.5, 4.9, 2.0)
    _arrow(ax, 7.4, 2.5, 8.3, 2.0)
    _arrow(ax, 5.8, 1.55, 7.4, 1.55)

    _box(ax, 11.0, 1.1, 1.7, 0.9, "Docker\nCompose", SLATE, size=8)
    _arrow(ax, 10.4, 1.55, 11.0, 1.55)

    fig.tight_layout()
    out = DOCS / "architecture_diagram.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def make_er():
    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.text(0.2, 7.3, "Vantara — PostgreSQL schema (ER diagram)", fontsize=13, fontweight="bold", color=NAVY)

    def table(x, y, w, title, cols):
        h = 0.42 + 0.26 * len(cols)
        ax.add_patch(mpatches.FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0.02",
                                             linewidth=1, edgecolor=NAVY, facecolor="white"))
        ax.add_patch(mpatches.Rectangle((x, y - 0.42), w, 0.42, facecolor=NAVY, linewidth=0))
        ax.text(x + 0.1, y - 0.28, title, color="white", fontsize=9, fontweight="bold")
        for i, c in enumerate(cols):
            ax.text(x + 0.1, y - 0.62 - 0.26 * i, c, fontsize=7.5, color="#1e293b", family="monospace")
        return (x, y, w, h)

    customers = table(0.4, 6.9, 2.6, "customers", ["customer_id  PK", "primary_country", "first_seen"])
    preds = table(3.6, 6.9, 3.0, "predictions", [
        "id  PK",
        "customer_id  FK",
        "model_version",
        "calibrated_probability",
        "risk_tier",
        "predicted_value",
        "value_at_risk",
        "shap_top / features (JSON)",
    ])
    segs = table(7.2, 6.9, 2.4, "segments", ["id  PK", "customer_id  FK", "persona", "kmeans_cluster"])
    recs = table(10.0, 6.9, 2.3, "recommendations", ["id  PK", "customer_id  FK", "action", "channel", "priority_score"])
    table(0.4, 3.7, 3.0, "model_registry", ["id  PK", "model_version  UQ", "churn_model", "clv_model", "decision_threshold"])
    table(3.6, 3.7, 2.8, "prediction_log", ["id  PK", "customer_id", "model_version", "latency_ms", "logged_at"])
    table(6.8, 3.7, 2.6, "drift_reports", ["id  PK", "feature", "psi", "drifted", "batch_date"])
    cprod = table(9.8, 3.7, 2.5, "customer_products", ["id  PK", "customer_id  FK", "product_id", "quantity"])

    for t in (preds, segs, recs, cprod):
        _arrow(ax, customers[0] + customers[2], 6.4, t[0], 6.4, color=SLATE)

    fig.tight_layout()
    out = DOCS / "er_diagram.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def make_workflow():
    """End-to-end workflow: build the models, score and serve, then act and monitor."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-0.5, 14.2)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.text(0.15, 7.75, "Vantara — end-to-end workflow", fontsize=13, fontweight="bold", color=NAVY)

    def lane(y, h, title, subtitle):
        ax.add_patch(mpatches.FancyBboxPatch((0.1, y), 14.0, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                             linewidth=1, edgecolor="#CBD5E1", facecolor="#F8FAFC"))
        ax.text(0.25, y + h - 0.28, title, fontsize=10, fontweight="bold", color=NAVY)
        ax.text(0.25, y + h - 0.55, subtitle, fontsize=8, color=SLATE, family="monospace")

    # 1. Build: python -m src.pipeline all (src/pipeline.py run_all order).
    lane(5.15, 2.35, "1 · Build", "python -m src.pipeline all")
    stages = [
        ("Ingest\nUCI Online\nRetail II", NAVY),
        ("Clean\nflag returns,\ndrop no-ID", SLATE),
        ("Validate\npandera\ncontracts", SLATE),
        ("Snapshots\n10 monthly\ncutoffs", INDIGO),
        ("Train\nOptuna · calibrate\n6 classical + 3 DL", INDIGO),
        ("Segment\nK-Means\n+ GMM", INDIGO),
        ("Explain\nSHAP + LIME", INDIGO),
        ("Recommend\nitem-item\nproducts", INDIGO),
    ]
    x0, step, w = 0.35, 1.72, 1.5
    for i, (label, colour) in enumerate(stages):
        _box(ax, x0 + i * step, 5.4, w, 1.0, label, colour, size=7.5)
        if i:
            _arrow(ax, x0 + i * step - (step - w), 5.9, x0 + i * step, 5.9)
    ax.text(x0 + 3 * step + w / 2, 5.27, "point-in-time features: only data before each cutoff (leakage tests)",
            ha="center", fontsize=7, color=AMBER, style="italic")

    # Artifacts hand over from build to serving.
    _box(ax, 4.6, 4.35, 9.35, 0.55,
         "models_artifacts/: models · scaler · calibrator · recommender · model_card.json (MLflow-logged)",
         AMBER, size=8)
    _arrow(ax, x0 + 4 * step + w / 2, 5.4, x0 + 4 * step + w / 2, 4.9)
    _arrow(ax, x0 + 7 * step + w / 2, 5.4, x0 + 7 * step + w / 2, 4.9)

    # 2. Score and serve: docker compose up.
    lane(2.4, 1.75, "2 · Score and serve", "docker compose up --build")
    _box(ax, 0.35, 2.6, 2.4, 0.95, "Batch scorer\nrisk · 90-day value ·\nvalue at risk · action", INDIGO, size=8)
    _box(ax, 3.35, 2.6, 2.4, 0.95, "PostgreSQL\npredictions · segments ·\nrecommendations", NAVY, size=8)
    _box(ax, 6.35, 2.6, 2.4, 0.95, "FastAPI\n/predict · /customers ·\n/explain · /campaign/plan", EMERALD, size=8)
    _box(ax, 9.35, 2.6, 2.4, 0.95, "Streamlit dashboard\n(reads only the API)", EMERALD, size=8)
    _box(ax, 12.2, 2.6, 1.75, 0.95, "Retention &\nmarketing\nteams", SLATE, size=8)
    _arrow(ax, 4.9, 4.35, 2.45, 3.55)
    _arrow(ax, 8.0, 4.35, 7.55, 3.55)
    for x in (2.75, 5.75, 8.75, 11.75):
        _arrow(ax, x, 3.07, x + 0.6 if x < 11 else x + 0.45, 3.07)

    # 3. Act and monitor.
    lane(0.15, 2.0, "3 · Act and monitor", "dashboard views · /metrics")
    _box(ax, 9.35, 0.35, 2.4, 0.95, "Campaign planner\nbudget × assumed\nsave rate", INDIGO, size=8)
    _box(ax, 12.2, 0.35, 1.75, 0.95, "Contact list\nwin-back · loyalty ·\nservice desk", SLATE, size=7.5)
    _box(ax, 0.35, 0.35, 2.4, 0.95, "Drift report (PSI)\n+ prediction log", AMBER, size=8)
    _box(ax, 3.35, 0.35, 2.4, 0.95, "Model Health page\nPSI > 0.2 flags drift", AMBER, size=8)
    _box(ax, 6.35, 0.35, 2.4, 0.95, "Retrain\nre-run the pipeline\nwhen drift persists", NAVY, size=8)
    _arrow(ax, 10.55, 2.6, 10.55, 1.3)
    _arrow(ax, 11.75, 0.82, 12.2, 0.82)
    _arrow(ax, 2.35, 2.6, 2.35, 1.3)  # right of the lane title
    _arrow(ax, 2.75, 0.82, 3.35, 0.82)
    _arrow(ax, 5.75, 0.82, 6.35, 0.82)
    # Retraining loops back to the start of the pipeline, under lane 3 and up the margin.
    ax.plot([7.55, 7.55, -0.25, -0.25], [0.35, 0.24, 0.24, 5.9], color=AMBER, lw=1.2, linestyle="--")
    ax.annotate("", xy=(0.35, 5.9), xytext=(-0.25, 5.9),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.2, linestyle="--"))

    ax.text(14.1, 7.75, "CI on every push: ruff · pytest (coverage ≥ 70%)",
            ha="right", fontsize=8, color=EMERALD, fontweight="bold")

    fig.tight_layout()
    out = DOCS / "workflow_diagram.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    make_architecture()
    make_er()
    make_workflow()
