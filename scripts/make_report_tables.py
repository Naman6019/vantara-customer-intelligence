"""Regenerate the report comparison tables from MLflow and the model card (PRD §8.3).

Usage: python scripts/make_report_tables.py
Writes Markdown tables to docs/tables.md so the final report is never hand-typed.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_project_root  # noqa: E402


def _df_to_md(df: pd.DataFrame, index: bool = False) -> str:
    return df.to_markdown(index=index)


def main() -> None:
    root = get_project_root()
    art = root / "models_artifacts"
    out_dir = root / "docs"
    out_dir.mkdir(exist_ok=True)

    card = json.loads((art / "model_card.json").read_text(encoding="utf-8"))
    churn = pd.read_csv(art / "churn_model_comparison.csv")
    clv = pd.read_csv(art / "clv_model_comparison.csv")

    sections = []
    sections.append("## Selected model (from model_card.json)\n")
    sections.append(
        f"- Churn model: **{card['churn_model']}** (threshold {card['decision_threshold']:.4f})\n"
        f"- Value model: **{card['clv_model']}**\n"
        f"- Validation ROC-AUC: {card['validation_metrics'].get('roc_auc')} | "
        f"Test ROC-AUC: {card['test_metrics'].get('roc_auc')} | "
        f"Test recall: {card['test_metrics'].get('recall')}\n"
        f"- CLV validation MAE: £{card['clv_validation_metrics'].get('mae')} | "
        f"top-decile value capture: {card['clv_validation_metrics'].get('value_capture_top_decile')}\n"
    )
    sections.append("\n## Churn model comparison (validation-ranked)\n")
    sections.append(_df_to_md(churn))
    sections.append("\n\n## Value model comparison\n")
    sections.append(_df_to_md(clv))

    (out_dir / "tables.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {out_dir / 'tables.md'}")


if __name__ == "__main__":
    main()
