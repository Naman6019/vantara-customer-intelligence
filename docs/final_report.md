# VANTARA RETAIL SOLUTIONS
## Data Science & Analytics Division
### Final Technical Report — Customer Behavior Prediction Platform

> Rebuilt against the approach statement and PRD acceptance criteria, using
> **out-of-time** evaluation. Headline numbers are lower than the original draft
> because the original random split leaked future information; §10 explains the
> difference. All tables are regenerated from MLflow / the model card by
> `scripts/make_report_tables.py`.

---

## 1. Executive Summary & Business Context

Vantara Retail Solutions operates an omnichannel retailer serving consumers and
small resellers. The Retention and Marketing teams previously relied on reactive,
rule-based segments ("no order in 90 days") that treat every customer in a segment
identically. This platform replaces that with a customer-level decision system:

1. **Calibrated churn risk** — out-of-time test **ROC-AUC 0.756**, recall **75.8%**
   at the operating threshold (recall ≥ 0.70).
2. **90-day customer value** — **LightGBM Tweedie**, validation **MAE £284**;
   it captures **63.9%** of the future value in the top decile (baseline: 12.5%).
3. **Value-at-risk prioritisation** — the at-risk leaderboard is ranked by
   `calibrated P(churn) × predicted 90-day value`, available in the API and dashboard.
4. **Explainable retention recommendations** — a rule table maps (segment, risk tier,
   value tier, top SHAP drivers) to an action, each with a cited reason.
5. **Probabilistic CLV benchmark (Beyond PRD)** — BG/NBD + Gamma-Gamma, with P(alive),
   reported honestly against the supervised model.
6. **Budget-constrained targeting (Beyond PRD)** — a greedy knapsack planner that
   maximises expected retained value per £ spent, with a save-rate sensitivity grid.

**Stack:** Python 3.11, pandas, scikit-learn, XGBoost, LightGBM, PyTorch, SHAP, LIME,
Optuna, MLflow, pandera, lifetimes, FastAPI, SQLAlchemy 2, PostgreSQL 15, Streamlit,
Docker Compose, pytest, ruff, GitHub Actions.

---

## 2. Data Engineering & Leak-Free Architecture

- **Source:** UCI *Online Retail II* (`id=502`), 1,067,371 line items, 2009-12-01 → 2011-12-09.
- **Cleaning:** drop missing Customer ID for customer-level modelling, deduplicate exact
  lines, remove administrative stock codes; returns/cancellations are **flagged, not dropped**
  (return behaviour is a signal). Cleaned corpus cached to Parquet.
- **Point-in-time discipline:** for every cutoff `T`, features use only rows with
  `invoice_date < T`; labels use only the window `[T, T + 90d)`. This is asserted in tests.
- **Active cohort:** a customer enters the retention cohort only with a purchase in
  `[T − 365d, T)` (`active_lookback_days: 365`). Customers who left long ago are not a
  retention target and previously inflated the churn rate.

### 2.1 Stacked monthly snapshots (plan D1)

Ten monthly training cutoffs (2010-06-01 … 2011-03-01) are stacked, giving
**36,454 training rows** rather than ~3.5k; validation is the 2011-06-01 snapshot and the
test snapshot is 2011-09-01, **scored once**. Every training label window closes before the
validation cutoff, and every validation window before the test cutoff.

| Snapshot | Cutoffs | Rows | Churn rate |
|---|---|---|---|
| Train | 2010-06-01 … 2011-03-01 (10) | 36,454 | 41%–67% |
| Validation | 2011-06-01 | 4,325 | 63.4% |
| Test (once) | 2011-09-01 | 4,323 | 50.5% |

*Cutoff 2011-09-01:* observation rows 623,150; label-window rows 151,811; cohort 4,323.

---

## 3. Supervised Model Leaderboards

### 3.1 Churn classification (selection on the **validation** snapshot)

Hyperparameters were tuned with **Optuna (TPE)** scored by forward-chaining CV over the
training cutoffs only. The model is chosen on validation ROC-AUC; the test snapshot is
scored once.

| Model | Val ROC-AUC | Val Recall | Val Precision | Val F1 | CV ROC-AUC (fwd-chain) |
|:---|---:|---:|---:|---:|---:|
| **Logistic Regression (production)** | **0.7967** | 0.755 | 0.810 | 0.782 | 0.7815 |
| Random Forest | 0.7897 | 0.814 | 0.794 | 0.804 | 0.8347 |
| Decision Tree | 0.7873 | 0.804 | 0.800 | 0.802 | 0.7740 |
| XGBoost | 0.7617 | 0.808 | 0.770 | 0.788 | 0.8869 |
| LightGBM | 0.7602 | 0.781 | 0.781 | 0.781 | 0.8613 |
| KNN | 0.7562 | 0.816 | 0.770 | 0.792 | 0.7976 |

**Calibration and threshold (plan D4):** isotonic regression is fit on validation; the
operating threshold is the *highest-precision* threshold that keeps recall ≥ 0.70 →
**threshold 0.6709** (validation recall 0.708, precision 0.828). Risk tiers come from
quantiles of the calibrated probability, not hand-picked constants.

**Out-of-time test result (used once):** ROC-AUC **0.7559**, recall **0.7579**, precision
0.6722, F1 0.7125, accuracy 0.6914, calibrated Brier 0.2143. AUC is below the PRD's 0.80
target; §10 explains why and why the number is nonetheless trustworthy.

### 3.2 90-day value regression (selection on validation MAE + value capture)

Every model must beat the **momentum baseline** (spend in the last 90 days). Selection is
on validation MAE, tie-broken by top-decile value capture.

| Model | Val R² | Val MAE (£) | Val RMSE (£) | Top-decile value capture |
|:---|---:|---:|---:|---:|
| **LightGBM Tweedie (production)** | 0.7543 | **283.76** | 1065.09 | **0.6391** |
| Random Forest | 0.7986 | 314.04 | 964.24 | 0.6386 |
| Spend Baseline (last 90d) | −0.0300 | 372.06 | 2180.61 | 0.1253 |
| Ridge (log1p) | −2.6231 | 442.34 | 4089.81 | 0.5840 |

The production model beats the baseline by ~24% MAE and captures **5×** more value in the
top decile. Out-of-time test: R² 0.4887, MAE £437.94, top-decile capture 0.5798.

### 3.3 Probabilistic CLV benchmark (Beyond PRD P1)

BG/NBD (order frequency) + Gamma-Gamma (spend per order) on the same observation window,
with **P(alive)** per customer:

| Model | Customers | MAE (£) | Top-decile capture |
|:---|---:|---:|---:|
| **Supervised LightGBM Tweedie** | 3,456 | ~284 | ~0.64 |
| BG/NBD + Gamma-Gamma | 1,656 | 524.92 | 0.403 |

**Honest negative:** the textbook probabilistic model does **not** beat the supervised
model here, because the cohort is small and the 90-day horizon is short. Its value is the
independent **P(alive)** view, shown in Customer 360.

---

## 4. Deep Learning Comparative Analysis (out-of-time test snapshot)

| Architecture | Objective / Input | Test metric | Finding |
|:---|:---|:---|:---|
| Feed-Forward ANN | Tabular features, BatchNorm, Dropout(0.25), early stopping | ROC-AUC **0.7492**, recall **0.9124** | Highest recall; useful as an ensemble consensus check. |
| Sequence LSTM | Ordered `[gap_days, spend, qty]` per customer | ROC-AUC **0.7765**, recall **0.9052** | Learns temporal cadence beyond static features; matches the best classical model. |
| Spending Autoencoder | Unsupervised reconstruction, 17→12→4→12→17 | threshold **0.7487** (95th pct) | Flags ~5% of accounts for manual review. |

---

## 5. Customer Segmentation & Persona Profiles

K-Means (k=4, silhouette **0.4618**, Davies-Bouldin **0.7585**) with GMM as the
alternative-algorithm check (silhouette 0.1603). Personas are assigned by a rank-based
rule so no two clusters collapse to the same label.

| Persona | n | Avg recency | Avg freq | Avg spend | Churn rate | Recommended play |
|:---|---:|---:|---:|---:|---:|:---|
| High-Value Champions (VIP) | 20 | 24 d | 126.3 | £134,398 | 5% | Loyalty / early access, no discount |
| Loyal & Active Frequent Buyers | 2,328 | 56 d | 8.8 | £3,552 | 34% | Cross-sell, category expansion |
| High-Return Service-Recovery | 57 | 177 d | 3.7 | £1,739 | 72% | Service-recovery outreach, no discount |
| Dormant / Low-Engagement | 1,918 | 251 d | 2.5 | £755 | 71% | Automated win-back email |

---

## 6. Retention Decision System

### 6.1 Value at risk (plan D4)
`value_at_risk = calibrated P(churn) × predicted 90-day value`. The batch scorer persisted
**4,269 scored customers**, total value at risk **≈ £412,835**.

### 6.2 Recommendation engine (plan D5)
`config/recommendations.yaml` maps (risk tier, value tier, top SHAP drivers, segment) to an
action. First match wins; every action cites its trigger. There is **no treatment/outcome
history**, so learning which intervention works (uplift modelling) is impossible — the rules
encode policy, and §10 proposes the A/B test to replace the assumed save rate with a measured one.

### 6.3 Budget-constrained targeting (Beyond PRD P3)
Given budget `B`, channel costs `c`, and an assumed save rate `s`, a greedy knapsack selects
the customers maximising `Σ p·value·s − c` subject to `Σ c ≤ B`. Greedy is validated against
a brute-force optimum in the tests. Because `s` is unmeasured, the API returns a **sensitivity
grid** over `s ∈ {5%, 10%, 20%}` and a four-policy comparison (top-risk / top-spender /
rule-table / optimised). *Save-rate values are assumptions and are labelled as such.*

---

## 7. Model Explainability (XAI)

- **Global:** SHAP summary in `docs/shap_global_summary.png`; recency and engagement dominate.
- **Local:** per-customer SHAP **and LIME** are served by the API
  (`GET /customers/{id}/explain`) on the served model's stored feature vector, and shown
  side by side in the dashboard (PRD acceptance criterion 4).
- **Plain language:** a narrative template translates the drivers into a sentence a marketing
  analyst can read, e.g. *"Calibrated churn risk 43.9% (Moderate). Top warning sign: recency."*

---

## 8. Deployment, Monitoring & Reproducibility

- **One command:** `docker compose up --build` starts PostgreSQL, a one-off **scorer**
  (populates predictions/segments/recommendations), the **API** (:8000) and the
  **Streamlit dashboard** (:8501). Credentials come from `.env` (no hard-coded secrets).
- **Pipeline:** `python -m src.pipeline all` runs ingest → clean → validate → snapshots →
  train → segment → explain → score. `dvc.yaml` declares the same stages.
- **Serving:** the API loads artifacts in a FastAPI `lifespan` handler and reads its model,
  feature list and threshold from `models_artifacts/model_card.json` (no hard-coded algorithm).
- **Monitoring (plan D7):** `/metrics` (Prometheus) for service health; a per-feature **PSI
  drift report** vs the training reference; and delayed-label performance from the backtest.
  The December batch flags `frequency` (PSI 0.47), `interpurchase_interval_std` (0.45) and
  `purchase_frequency_trend` (0.38) — expected, since the Christmas trading window shifts cadence.
- **Quality gates:** ruff clean, mypy configured, **pytest with 75% coverage** on `src/`+`api/`
  (PRD requires ≥ 70%), GitHub Actions CI on every push.

---

## 9. Mathematical Foundations Appendix

### 9.1 Logistic Regression & Binary Cross-Entropy
`P(y=1|x) = σ(wᵀx + b)`, minimising
`L = −(1/N) Σ [yᵢ log ŷᵢ + (1−yᵢ) log(1−ŷᵢ)] + (λ/2)‖w‖₂²`.

### 9.2 Isotonic Probability Calibration
Calibration finds a monotone `g` minimising `Σ (yᵢ − g(ŷᵢ))²` subject to `g` non-decreasing,
so that `P̂(churn)` can be multiplied by value to give an interpretable expected loss (Brier score).

### 9.3 Value-at-Risk & Budget-Constrained Targeting
`VaRᵢ = p̂ᵢ · v̂ᵢ`. The campaign objective is the 0/1 knapsack
`max Σ xᵢ (s·VaRᵢ − cᵢ) s.t. Σ xᵢ cᵢ ≤ B`, solved by the greedy ratio
`(s·VaRᵢ − cᵢ)/cᵢ`; exact for equal costs and near-optimal otherwise, with a brute-force check.

### 9.4 BG/NBD + Gamma-Gamma (probabilistic CLV)
BG/NBD models the drop-out and purchase process; the conditional expectation of purchases in
`(0, t]` is `E[X(t) | x, tₓ, T]`. Gamma-Gamma models the average order value. Their product is
the expected value; `P(alive)` is the posterior drop-out survival probability.

### 9.5 K-Means & Silhouette
`argmin_S Σⱼ Σ_{x∈Sⱼ} ‖x−μⱼ‖²`; `s(i) = (b(i)−a(i)) / max(a(i), b(i))`.

### 9.6 Evaluation Metrics
Precision `TP/(TP+FP)`, Recall `TP/(TP+FN)`, `F1 = 2PR/(P+R)`,
`ROC-AUC = integral  TPR d(FPR)`, and top-decile value capture `= (Σ value in top-ranked 10%) / (Σ value)`.

---

## 10. Why the numbers changed (random split → out-of-time)

The original draft used a single snapshot, a **random** 70/15/15 split, a 99-day window and
an all-history cohort. That let train and test customers come from the same period and made
AUC and R² look better than the model really is. The corrected, honest comparison:

| Dimension | Original draft | Corrected (this report) |
|:---|:---|:---|
| Evaluation | Random stratified split (same period) | Out-of-time by cutoff; test used once |
| Churn test ROC-AUC | 0.7846 (XGBoost) | **0.7559** (Logistic Regression, selected on validation) |
| Churn recall | 81.3% | **75.8%** at the recall-constrained threshold |
| CLV test R² | 0.8447 (Ridge) | **0.4887** (LightGBM Tweedie); validation 0.7543 |
| Window | ~100 days | **90 days** (config-driven) |
| Cohort | 5,225 (all ever seen) | **4,323** active within 365 days |
| Churn rate | 55.25% | **50.45%** |

**Documented deviations and negatives (PRD §20.7):**
1. Out-of-time AUC is below the 0.80 target. This is expected: the task is harder when the
   model is scored on a later period, and the cohort excludes long-departed customers who
   made the old problem easy. The lower number is the honest one.
2. The original Ridge "R² 0.84" was a **leakage artifact** — `historical_clv` duplicated
   `monetary_total` and random splitting let a linear model exploit it. `historical_clv` was
   removed; the value model now beats a real baseline.
3. BG/NBD + Gamma-Gamma does not beat the supervised value model here (reported above).
4. Campaign save rates are assumptions; an A/B test design is the next step.
5. No real intervention data → no uplift modelling.

---

## 11. Future Work

- Transformer sequence model (deferred by PRD §4.2).
- Real-time ingestion (Kafka/event pipeline) instead of scheduled batch scoring.
- Role-based access control and multi-tenant auth.
- Uplift modelling once A/B treatment data exists; measure the true save rate.
- Extend the shared engineering `template/` (P5) into FlowCast and Northgate.

---

## 12. Verification Summary

| Check | Result |
|:---|:---|
| `docker compose up --build` on a clean clone | API :8000 + dashboard :8501 + populated DB |
| Out-of-time evaluation, test used once | Yes (asserted in `tests/test_data_split.py`) |
| 90-day windows enforced & tested | Yes (`tests/test_features.py`) |
| At-risk ranked by calibrated P(churn) × value | Yes API + dashboard |
| SHAP **and** LIME per customer in the dashboard | Yes `/customers/{id}/explain` |
| `/metrics`, drift report, Model Health page | Yes |
| Test coverage on `src/` (+`api/`) ≥ 70% | Yes 75% |
| Final report PDF with regenerated tables | Yes `docs/final_report.pdf` |

— *Vantara Retail Solutions Data Science Division* —
