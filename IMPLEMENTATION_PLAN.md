# Vantara Customer Intelligence Platform — Completion Plan

> **Project:** PROJ-VANT-260704 · Artificial Intelligence · Priority **High**
> **Assigned:** 20 Aug 2026 · **Course ends:** 20 Nov 2026 · **Plan written:** 26 Sep 2026
> **Portal status:** In progress, 0%, **nothing uploaded yet**
> **Spec sources:** `extracted_requirements.md` (PRD v1.0) + the approach statement submitted on the portal (quoted below)

**Goal:** Close the gap between what is built and what was promised, so the submission matches the approach statement and meets the PRD acceptance criteria.

**Current architecture:** A Pandas feature pipeline with a point-in-time cutoff, then sklearn/XGBoost/LightGBM/PyTorch models saved as joblib/pt files. FastAPI loads those files at import time. The Streamlit dashboard loads the same files directly. A PostgreSQL container is declared in Docker Compose but nothing uses it.

**Target architecture:** A single-command pipeline that builds multi-cutoff snapshots and evaluates on later time periods (out-of-time). A batch scoring job writes scores, segments and recommendations to PostgreSQL. FastAPI reads from PostgreSQL. The Streamlit dashboard talks only to the API. Monitoring comes from `/metrics` and drift reports.

**Tech stack (unchanged):** Python 3.11, pandas, scikit-learn, XGBoost, LightGBM, PyTorch, SHAP, LIME, FastAPI, SQLAlchemy 2, PostgreSQL 15, Streamlit, Docker Compose, pytest, ruff. **New:** `mlflow` (local file store), `prometheus-fastapi-instrumentator`, `pytest-cov`.

---

## 1. The promise we are delivering against

The approach statement you submitted on the portal is what reviewers will grade against:

> *"…predicts churn risk and estimated **90 day** customer value from historical transaction data. The system will **segment customers, prioritize high value at risk customers**, and provide **explainable retention recommendations through a dashboard and REST API**. I will use **leakage safe time based model evaluation**, deploy the application with **Docker**, and include **monitoring, testing, and documentation**…"*

Every bold phrase needs to be visibly true in the code, the dashboard and the final report.

---

## 2. Current state (audited 26 Sep 2026)

### What's done and solid
| Area | Evidence |
|---|---|
| Ingestion + cleaning + validation | `src/data/loader.py`, `cleaner.py`, `validator.py`; returns flagged, not dropped |
| Point-in-time feature builder | `src/features/builder.py` — features built only from rows before the cutoff (`< cutoff`), targets only from rows on or after it (`>= cutoff`) |
| 6 classical churn models + 5 CLV regressors | `src/models/churn_models.py`, `clv_regressor.py`, artifacts in `models_artifacts/` |
| ANN, LSTM, Autoencoder | `src/models/dl_models.py`, `.pt` artifacts |
| K-Means + GMM with personas | `src/segmentation/clusterer.py`, `kmeans_k_metrics.csv` (k=4, silhouette 0.46) |
| SHAP / LIME / PDP / narrative | `src/explainability/explainer.py`, `docs/shap_global_summary.png`, `docs/pdp_plots.png` |
| FastAPI with `/health`, `/metadata`, `/predict`, `/batch-score` | `api/routers/predictions.py` |
| Streamlit dashboard (6 views) | `frontend/dashboard.py`, screenshots in `docs/` |
| ~75 pytest tests incl. leakage tests | `tests/` |
| Final report draft | `docs/final_report.md` |

### Gaps (ordered by priority)

**P0 means a promise in the approach is broken or an acceptance criterion fails.** P1 is a correctness or credibility problem a reviewer is likely to notice. P2 is polish.

| # | Gap | Evidence | Why it matters | Pri |
|---|---|---|---|---|
| G1 | **Evaluation is not time-based.** The code takes one snapshot and splits it randomly (stratified 70/15/15). | `src/models/data_split.py:76-90` | This directly contradicts "leakage safe time based model evaluation". Train and test customers come from the same period, so the reported scores don't show how the model does on later data. | P0 |
| G2 | **The prediction window is 99 days, not 90.** The window runs from the cutoff to the end of the data (2011-09-01 → 2011-12-09). `churn_window_days: 90` in the config is never read. | `builder.py:19-20`, `config.yaml:56` | The approach says "90 day customer value". The report says "100 days". | P0 |
| G3 | **The best model is chosen on the test set.** The validation split is never used, and hyperparameters are fixed with no tuning. | `churn_models.py:188-192`, `data_split.py` returns `X_val` but nothing reads it | Breaks PRD §8.3 ("test set touched exactly once") and makes the test score optimistic. | P0 |
| G4 | **PostgreSQL is never used.** No SQLAlchemy code exists, only the Docker Compose service. | grep `create_engine`, `sqlalchemy` → only `requirements.txt` | PRD §12.1 and §14 require stored scores, and the architecture diagram shows a database. | P0 |
| G5 | **The dashboard bypasses the API.** It loads joblib files directly and reads `data/interim/*.parquet`, which `.dockerignore` excludes. | `dashboard.py:294`, `dashboard.py:752`, `.dockerignore` | The approach says recommendations come "through a dashboard **and** REST API". The revenue page breaks inside Docker. | P0 |
| G6 | **Retention recommendations exist only in the dashboard,** as hard-coded if/else rules on churn probability and total spend. There is no "value at risk" ranking. | `dashboard.py:316-329` | Misses "prioritize high value at risk customers" and "explainable recommendations … through REST API". | P0 |
| G7 | **No monitoring at all.** | grep `metrics`, `drift`, `prometheus` → nothing | Monitoring is an explicit promise in the approach. | P0 |
| G8 | **Not under version control, and nothing uploaded.** | no `.git`; portal shows 0% | Any mistake is unrecoverable, and the reviewer sees no progress. | P0 |
| G9 | **The cohort includes customers who left long ago.** Everyone ever seen before the cutoff is included, e.g. someone last active 568 days earlier. | `builder.py:39`, report §6.2 | This inflates the churn rate (55%) and makes AUC easy for no business reason. You can't "retain" someone who left 18 months ago. | P1 |
| G10 | **The deployed model is not the "best" one.** The API hard-codes XGBoost while `best_churn_model.joblib` is Logistic Regression, and the metrics in `/metadata` are hard-coded. | `predictions.py:28-31`, `predictions.py:59-60` | The model you serve, the metrics you report and the model you selected all disagree. | P1 |
| G11 | **The decision threshold is fixed at 0.5 and the risk tiers are magic numbers.** | `predictions.py:79-80` | The "81% recall" claim depends on the threshold. Ranking by value at risk also needs calibrated probabilities. | P1 |
| G12 | **The engagement score uses percentile ranks across the population.** It's computed over train and test together, and at inference it depends on whichever batch is being scored. | `builder.py:260-262` | Minor leakage, and a single customer's score can't be reproduced in the API. | P1 |
| G13 | **The CLV result looks suspicious.** Ridge gets R² 0.84 while the tree models get 0.39–0.67. `historical_clv` is just `monetary_total` clipped at 0, so the two features are perfectly collinear. | `builder.py:103`, `clv_model_comparison.csv` | Heavy-tailed targets with a few very large customers let a linear model look great by chance. There's no simple baseline to compare against. | P1 |
| G14 | **The `notebooks/` folder is empty,** but the README lists 3 notebooks. | `notebooks/` | Required deliverables in PRD §15. | P1 |
| G15 | **Credentials are hard-coded in Docker Compose.** | `docker-compose.yml:24,32` | Breaks the PRD §13 security requirement. | P1 |
| G16 | **Two frontends:** a React template (`frontend-crm/`, shadcn-admin showing static `crm-data.json`) and the Streamlit dashboard. | `frontend-crm/src/data/crm-data.json`, `.dockerignore` says "Unused React CRM prototype" | Confusing for reviewers, and roughly 300 MB of `node_modules` sitting in the project. | P1 |
| G17 | **LIME is not shown in the dashboard.** | no `lime` match in `dashboard.py` | Acceptance criterion 4: "SHAP **and LIME** … through the dashboard". | P2 |
| G18 | **Test coverage is unknown.** `coverage` isn't installed in the active environment and `.coverage` is stale. | `python -m coverage` → not installed | Acceptance criterion 6 requires ≥70% coverage. | P2 |
| G19 | **Loose files:** duplicate `.docx` files (`9d04…docx` and `scratch_doc.docx`, same size), and the final report is Markdown rather than the PDF required in §15. | project root | Housekeeping. | P2 |

**Expect the headline numbers to drop after G1–G3 and G9 are fixed.** Out-of-time evaluation on customers who are still active is harder than the current setup. That lower number is the honest one. The report should show both side by side and explain the difference. A reviewer will trust that more than a number that looks too good.

---

## 3. Key decisions

### D1 — Time-based evaluation with stacked monthly snapshots (fixes G1, G3)
**Options:**
- (a) Keep the random split.
- (b) Use one snapshot and split customers by cohort.
- (c) **Build multiple cutoff snapshots and split by time.**

**Choice: (c).** For each cutoff T, features come from data before T and the label comes from [T, T+90d). A split is valid only if every training label window ends before the validation cutoff, and every validation label window ends before the test cutoff.

| Split | Cutoffs | Label windows end | Purpose |
|---|---|---|---|
| Train | 2010-06-01 … 2011-03-01 (monthly, 10 snapshots) | ≤ 2011-05-30 | Fitting and forward-chaining CV |
| Validation | 2011-06-01 | 2011-08-30 | Model selection, tuning, threshold, calibration |
| Test | 2011-09-01 | 2011-11-30 | **Touched once**, for final reporting |
| Production refit | train + 2011-04-01 … 2011-06-01 | ≤ 2011-08-30 | Final model saved as the artifact |
| Live scoring | as of 2011-12-09 (last date in the data) | — (unlabelled) | What the dashboard shows as "today" |

**Why this design:**
- It reproduces how the model is actually used: train on the past, score the future.
- It gives about 10× more training rows.
- It still keeps PRD §8.3's rule that the test set is used once.

The PRD's "stratified 70/15/15" is a *recommendation*, so the report should record this change as a deliberate, justified deviation. CV inside the training data uses **forward-chaining over cutoffs**, not shuffled k-fold, because the same customer appears in several snapshots.

### D2 — Cohort = customers active in the 365 days before the cutoff (fixes G9)
Customers with no purchase in the 12 months before T are excluded. The window is configurable as `features.temporal.active_lookback_days: 365`. They aren't a retention target, and including them inflates AUC. The report states the cohort size and churn rate at each cutoff.

### D3 — A 90-day value model that has to beat a simple baseline (fixes G2, G13)
- Target: net spend in [T, T+90d), clipped at 0.
- Drop `historical_clv`, because it's a copy of `monetary_total`.
- Add a **baseline**: spend in the last 90 days before T. Every model must beat it.
- Candidates:
  - Ridge on a `log1p` target
  - LightGBM with a `tweedie` objective, which fits zero-inflated, heavy-tailed spend well
  - Random Forest
- Selection metric on validation:
  - MAE
  - **Top-decile value capture:** the share of actual future value held by the top 10% of customers the model ranks highest. This is the metric the retention team cares about.
  - R² is reported but is not the selection criterion, because a few very large customers can swing it.

### D4 — Rank customers by expected value at risk, using calibrated probabilities (fixes G6, G11)
`value_at_risk = P(churn) × predicted_90d_value`, where P(churn) is **calibrated** with isotonic regression fit on the validation snapshot. The at-risk leaderboard is sorted by `value_at_risk`, which answers "prioritize high value at risk customers" literally.

The operating threshold is chosen on validation as the **highest-precision threshold that keeps recall ≥ 0.70** (the PRD target). It's saved in the model card. Risk tiers come from quantiles of calibrated probability, not magic numbers.

### D5 — Rule-based recommendation engine driven by explanations, served by the API (fixes G6)
There is no treatment or outcome history in the data, so learning which intervention works (uplift modelling) isn't possible. **Say this in the report.** Instead, `src/recommendations/engine.py` maps (segment, risk tier, value tier, top 3 SHAP drivers) to an action, and every action cites the trigger that fired it. Examples:

| Trigger | Action | Reason shown to the user |
|---|---|---|
| high value tier + high risk + `recency` among top drivers | Account manager call within 7 days | "No order for 74 days vs usual 21-day cycle" |
| `return_rate` among top drivers | Service-recovery outreach, no discount | "Return rate 18% (segment avg 4%)" |
| `discount_sensitivity` high + mid value | Targeted 10% voucher | "62% of items bought below median price" |
| low value + high risk | Automated win-back email | "Low value; automated channel only" |
| low risk + high value | Loyalty/early-access, no discount | "Healthy; avoid margin leakage" |

Rules live in `config/recommendations.yaml` so marketing can edit them without touching code.

### D6 — Batch scoring into PostgreSQL; the API reads from it; the dashboard uses only the API (fixes G4, G5)
- `src/serving/batch_score.py` computes features as of the latest date, then scores churn, value, segment, anomaly flag, SHAP top drivers and recommendations. It writes everything to PostgreSQL with a `model_version`.
- Tables: `customers`, `customer_features`, `predictions`, `segments`, `recommendations`, `model_registry`, `prediction_log`, `drift_reports`. ER diagram regenerated.
- The API gains `GET /customers/{id}`, `GET /customers/at-risk?limit=&segment=&country=`, `GET /segments`, `GET /revenue/monthly`, and `GET /monitoring/drift`. `/predict` and `/batch-score` stay for ad-hoc input and also write to `prediction_log`.
- The dashboard replaces every `joblib.load` and `read_parquet` call with `httpx` calls to `API_URL`.

**Why:** single-customer reads become simple primary-key lookups, which easily meets the p95 < 400 ms target. The dashboard can't drift out of sync with the served model. This is also the architecture the PRD §14 diagram describes.

### D7 — Monitoring you can demo without a Prometheus/Grafana stack (fixes G7)
- **Service health:** `prometheus-fastapi-instrumentator` exposes `/metrics` (request count, latency histogram, errors). The dashboard's "Model Health" page reads it and shows p95 latency.
- **Data drift:** after each batch run, compute PSI per feature between the training reference distribution (saved at training time) and the current batch. Store it in `drift_reports`. Flag PSI > 0.2.
- **Delayed-label performance:** because labels arrive 90 days late, show AUC, recall and value capture per cutoff over time from the backtest. This is what monitoring looks like once real outcomes come in.
- **Prediction log:** every API prediction is written with its model version and latency.
- Grafana is left out. It's another container and more configuration for no extra grading value.

### D8 — The model card is the single source of truth (fixes G10)
`models_artifacts/model_card.json` records:
- model version (git SHA plus date)
- training cutoffs and feature list
- threshold and calibrator path
- validation and test metrics
- the reference-distribution path

`/metadata`, the batch scorer and the report all read from it. The API loads whatever model the card names, so there's no hard-coded XGBoost.

### D9 — Experiment tracking with MLflow's local file store
`mlflow` with `file:./mlruns`, no server. Every training run logs parameters, metrics and artifacts. `scripts/make_report_tables.py` regenerates the comparison tables from MLflow, as PRD §8.3 requires. MLflow is chosen over a hand-written CSV log because it's the industry-standard tool and supports the "production-oriented AI engineering workflow" claim.

### D10 — Engagement score from saved training quantiles (fixes G12)
Fit the percentile breakpoints for recency, frequency and monetary on the training snapshots only, and save them in `engagement_reference.json`. At inference, score against those saved breakpoints. A single customer's score is then reproducible and doesn't depend on the batch it's scored with.

### D11 — Frontend: Streamlit is the official deliverable (G16) — **your call**
**Recommendation:** Streamlit stays the graded dashboard. It's what's in Docker Compose, it's the PRD's primary recommendation, and it's already built. Move `frontend-crm/` out of the repo, or delete `node_modules`/`dist` and mention it in the README as an exploratory prototype.

**Alternative:** if you'd rather present the React UI, it must call the API (not `crm-data.json`) and get its own Dockerfile. Budget about 3 extra days.

### D12 — One command, and continuous integration
- `python -m src.pipeline all` runs: load → clean → validate → snapshots → train → evaluate → calibrate → segment → explain → batch-score → drift.
- A `Makefile` (or `tasks.ps1` on Windows) wraps it.
- A GitHub Actions workflow runs `ruff` and `pytest --cov` on every push. It's cheap, and it proves "testing" to a reviewer.

---

## 4. Target repository changes

```
Vantara/
├── config/
│   ├── config.yaml              # + active_lookback_days, snapshot cutoffs, split dates, psi threshold
│   └── recommendations.yaml     # NEW rule table (D5)
├── src/
│   ├── pipeline.py              # NEW single-command orchestrator (D12)
│   ├── features/
│   │   ├── builder.py           # MODIFY: 90-day window, active cohort, drop historical_clv
│   │   ├── snapshots.py         # NEW: build_snapshots(cutoffs) -> stacked frame with `cutoff_date`
│   │   └── engagement.py        # NEW: fit/apply saved quantile reference (D10)
│   ├── models/
│   │   ├── data_split.py        # REWRITE: time-based split by cutoff (D1)
│   │   ├── churn_models.py      # MODIFY: select on validation, tune, MLflow logging
│   │   ├── clv_regressor.py     # MODIFY: baseline, tweedie LGBM, value-capture metric
│   │   ├── calibration.py       # NEW: isotonic calibrator + threshold search (D4)
│   │   └── model_card.py        # NEW (D8)
│   ├── recommendations/engine.py  # NEW (D5)
│   ├── serving/
│   │   ├── db.py                # NEW: SQLAlchemy engine/session from DATABASE_URL
│   │   ├── models.py            # NEW: ORM tables (D6)
│   │   └── batch_score.py       # NEW (D6)
│   └── monitoring/drift.py      # NEW: PSI + reference distributions (D7)
├── api/
│   ├── main.py                  # MODIFY: instrumentator, lifespan-based artifact loading
│   └── routers/
│       ├── predictions.py       # MODIFY: model from card, calibrated prob, threshold, log to DB
│       ├── customers.py         # NEW: /customers/{id}, /customers/at-risk, /segments
│       └── monitoring.py        # NEW: /monitoring/drift, /revenue/monthly
├── frontend/dashboard.py        # MODIFY: API-only data access, LIME panel, Model Health page
├── notebooks/01_eda.ipynb, 02_feature_engineering.ipynb, 03_model_experiments.ipynb  # NEW
├── scripts/make_report_tables.py  # NEW
├── .github/workflows/ci.yml     # NEW
├── .env.example                 # NEW (DATABASE_URL, POSTGRES_*, API_URL, LOG_LEVEL)
└── docs/final_report.pdf        # regenerated
```

---

## 5. Task plan (about 9 working days)

Each task ends in a state you can test yourself. Commit after every task. **Upload a zip or repo link to the portal at the end of each day marked 📤.**

### Task 0 — Safety net (½ day) 📤
- `git init`. Add `.gitattributes`. Confirm `.gitignore` covers data and artifacts. Commit. Push to a private GitHub repo.
- Delete `scratch_doc.docx`, since it's identical in size to the brief. Keep the original `.docx` in `docs/brief/`.
- `pip install pytest-cov coverage` and run `pytest --cov=src` to record today's baseline coverage in the commit message.
- Upload the current state to the portal so progress stops showing 0%.
- **Done when:** the repo is on GitHub, the baseline test run is green, and the portal shows a submission.

### Task 1 — Fix the target definition and the cohort (1 day)
**Files:** `src/features/builder.py`, `config/config.yaml`, `tests/test_features.py`
- Limit the prediction window to `[T, T + churn_window_days)` and read the length from config.
- Restrict the cohort to customers with a purchase in `[T - active_lookback_days, T)`.
- Drop `historical_clv`.
- Move engagement scoring to `src/features/engagement.py` with `fit_reference(df) -> dict` and `score(df, ref) -> Series`.
- **Tests to add:**
  - A transaction at T+90 days is excluded from the label.
  - A customer inactive for 400 days before T is not in the cohort.
  - Scoring one row against a saved reference gives the same value as scoring it inside a batch.
- **Done when:** the new tests pass and the log shows the cohort size and churn rate at T = 2011-09-01.

### Task 2 — Snapshot builder and time-based split (1 day)
**Files:** `src/features/snapshots.py` (new), `src/models/data_split.py` (rewrite), `config/config.yaml`, `tests/test_data_split.py`
- `build_snapshots(df_clean, cutoffs: list[str]) -> DataFrame` with a `cutoff_date` column.
- `time_split(snapshots, cfg) -> dict` returning train/val/test by cutoff, as in the D1 table.
- Fit the scaler on the training partition only.
- **Tests to add:**
  - The maximum training label end is before the validation cutoff.
  - The maximum validation label end is before the test cutoff.
  - No `(customer_id, cutoff_date)` pair appears twice.
  - The scaler's mean equals the training-only mean.
- **Done when:** the split tests pass and the old random-split tests are deleted or rewritten.

### Task 3 — Retrain with proper selection, calibration and MLflow (1.5 days) 📤
**Files:** `churn_models.py`, `clv_regressor.py`, `calibration.py` (new), `model_card.py` (new), `dl_models.py` (point at the new split)
- Tune the 6 classical models plus the ANN with a small random search, scored by forward-chaining CV over training cutoffs. Select on **validation** ROC-AUC, then recall at the chosen threshold.
- Retrain the LSTM and the Autoencoder on the new split.
- Calibrate the chosen churn model (isotonic, fit on validation) and pick the threshold as in D4.
- CLV: add the baseline and the tweedie LightGBM, and select on validation MAE plus top-decile capture.
- Refit on the production-refit cutoffs, save artifacts and write `model_card.json`.
- Evaluate on the **test** cutoff exactly once. Log everything to MLflow.
- **Tests:** `select_best()` never receives test data (assert on its function signature). The threshold search returns recall ≥ target on a synthetic example. The model card round-trips.
- **Done when:** `mlruns/` has a run for every model, `model_card.json` exists, and the test metrics are recorded. If ROC-AUC is below 0.80 on test, note it for the report discussion (PRD §20.7 allows this if it's explained).

### Task 4 — Recommendation engine (½ day)
**Files:** `src/recommendations/engine.py`, `config/recommendations.yaml`, `tests/test_recommendations.py`
- `recommend(row: dict, shap_top: list[dict], card: dict) -> {"action", "channel", "reasons": [str], "priority_score"}`
- **Tests:** one test per rule in the D5 table, plus a fallback rule for inputs that match nothing.
- **Done when:** every customer in a batch gets exactly one action with at least one reason.

### Task 5 — PostgreSQL storage and batch scoring (1.5 days) 📤
**Files:** `src/serving/{db,models,batch_score}.py`, `docker-compose.yml`, `.env.example`, `tests/test_batch_score.py`
- SQLAlchemy 2 ORM with the tables from D6, created with `Base.metadata.create_all` at startup. Alembic would be overkill for one schema version.
- The batch job builds features as of 2011-12-09, then scores, explains (top 3 SHAP drivers per customer stored as JSON), segments, flags anomalies, recommends and upserts.
- Move credentials out of Docker Compose into `.env`, referenced with `${POSTGRES_PASSWORD}`.
- Add a one-off `scorer` service to Docker Compose (`command: python -m src.serving.batch_score`) that runs after `db` is healthy. The API depends on it having completed successfully.
- **Tests:** run against SQLite in-memory (the same ORM works) and assert row counts and that `value_at_risk` = p × value.
- **Done when:** `docker compose up` gives a populated `predictions` table, which you can check with `psql`.

### Task 6 — API upgrade (1 day)
**Files:** `api/main.py`, `api/routers/{predictions,customers,monitoring}.py`, `api/schemas/*`, `tests/test_api.py`
- Load artifacts in a FastAPI `lifespan` handler, not at import time. Tests can then swap in fakes.
- Add the new endpoints from D6. `/predict` uses the calibrated probability and the model-card threshold and returns recommendations. `/metadata` reads the model card.
- Add `/metrics` via the instrumentator. Write a row to `prediction_log` on every call.
- **Tests:**
  - `/customers/at-risk` is sorted by `value_at_risk` descending.
  - Unknown customer returns 404.
  - A malformed CSV returns 400 with the list of missing columns.
  - `/metrics` contains `http_requests_total`.
  - A latency smoke test: 200 sequential `/customers/{id}` calls with p95 < 400 ms.
- **Done when:** every API test passes and Swagger `/docs` shows all the endpoints.

### Task 7 — Drift monitoring (½ day)
**Files:** `src/monitoring/drift.py`, `tests/test_drift.py`
- `psi(ref: np.ndarray, cur: np.ndarray, bins=10) -> float`.
- Save the reference histograms at training time. The batch job writes one `drift_reports` row per feature.
- **Tests:** identical distributions give PSI ≈ 0. A shifted normal distribution gives PSI > 0.2.
- **Done when:** the API's `/monitoring/drift` returns per-feature PSI.

### Task 8 — Dashboard through the API, plus the missing views (1 day) 📤
**Files:** `frontend/dashboard.py`, optionally split into `frontend/pages/*.py`
- Remove every `joblib.load` and `read_parquet` call. Use an `api_client.py` wrapper built on `httpx` with `st.cache_data(ttl=60)`.
- **At-Risk Priority** view: ranked by value at risk, with filters for segment, country and value tier, and the recommended action plus reasons in each row.
- **Customer 360:** SHAP waterfall **and LIME** side by side, with a sentence on whether they agree (fixes G17).
- **Model Health:** p95 latency, request volume, a PSI table with flags, and a chart of AUC/recall per cutoff over time.
- **Revenue trends:** read from `/revenue/monthly` instead of the parquet file.
- **Done when:** the whole dashboard works inside `docker compose up` on a fresh clone, with no local files needed.

### Task 9 — Notebooks, housekeeping and CI (½ day)
- Three notebooks that *call* `src/` functions rather than duplicating logic:
  - EDA ending in a list of hypotheses
  - Feature justification and VIF check
  - Model experiments reading from MLflow
- Resolve D11: move or trim `frontend-crm/`.
- Add `.github/workflows/ci.yml` running `ruff check .` and `pytest --cov=src --cov-fail-under=70`.
- **Done when:** CI is green on GitHub and coverage is ≥ 70%.

### Task 10 — Report, diagrams and final hand-in (1 day) 📤
- Regenerate every table from MLflow with `scripts/make_report_tables.py`.
- Add a **"Why the numbers changed"** section to the report: random split vs out-of-time, all customers vs active cohort, 99 vs 90 days. Show a before/after table.
- Update the architecture diagram (add PostgreSQL, the batch scorer and monitoring) and the ER diagram (new tables).
- Add a limitations and future-work section:
  - no real intervention data, so no uplift modelling
  - Transformer sequence model
  - real-time ingestion
  - role-based access control
- Export `docs/final_report.pdf`. Rewrite the README quick start so `cp .env.example .env && docker compose up --build` is all you need.
- Record a 5-minute walkthrough video, as PRD §17 day 20 asks.
- **Done when:** a clean clone on another folder or machine works end to end, and the final upload is on the portal.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Out-of-time AUC comes in well below 0.80 | Expected. Report it honestly next to the old number. Try the ANN+XGBoost average (PRD §11 hints at uncorrelated errors). Recall ≥ 0.70 is still reachable through the threshold. |
| Stacked snapshots slow down training | ~10 × 3.5k rows is under 50k rows, which is trivial for tree models. Cap the random search at about 30 trials per model. |
| The Docker image gets too large (torch, shap) | Use the CPU-only torch wheel (`--index-url https://download.pytorch.org/whl/cpu`) and a multi-stage build. |
| OneDrive syncing `mlruns/` and `node_modules` | Add both to `.gitignore`, and consider moving the repo out of OneDrive, since sync often locks files that pytest or Docker need. |
| Running out of time | Priority order: Tasks 0 → 1 → 2 → 3 → 5 → 6 → 8 → 10 are the must-haves. Tasks 4, 7 and 9 can shrink to their minimum versions. |

## 7. Definition of done (maps to PRD §20 + approach)
- [ ] `docker compose up --build` on a clean clone serves the API (:8000) and dashboard (:8501), with PostgreSQL populated
- [ ] Evaluation is out-of-time with documented cutoffs; the test cutoff was used once; the 90-day windows are enforced and tested
- [ ] At-risk list ranked by calibrated P(churn) × predicted 90-day value, available in the API and the dashboard
- [ ] Every recommendation shows its reasons; SHAP and LIME are visible per customer in the dashboard
- [ ] `/metrics`, a drift report and a Model Health page all exist
- [ ] ≥ 70% coverage on `src/`, ruff clean, CI green
- [ ] Final report PDF with regenerated tables, a "why numbers changed" section, the maths appendix and updated diagrams
- [ ] Portal submission uploaded

---

## 8. Professional track (Beyond PRD)

> The shared rules, engineering template and time budget are in [`../PROFESSIONAL_TRACK.md`](../PROFESSIONAL_TRACK.md). **Start this only once §7 is fully ticked.** Everything here is labelled "Beyond PRD" in the report and dashboard. **Tier A** = P1, P3, P5 (≈ 4 days). P2, P4 and P6 are optional.

### P1 — Probabilistic CLV benchmark: BG/NBD + Gamma-Gamma (Tier A, ~1 day)
- **What:** fit the standard "buy-till-you-die" models on the same training cutoffs:
  - BG/NBD predicts how many orders each customer will place in the next 90 days
  - Gamma-Gamma predicts the average spend per order
  - together they give a probabilistic 90-day value

  Use `pymc_marketing.clv` (`BetaGeoModel`, `GammaGammaModel`); fall back to `lifetimes` if pinning becomes a problem. Also report **P(alive)** per customer, a churn view that comes from a probabilistic model rather than a classifier.
- **Why it's credible:** this is the textbook industry baseline for customer value. Beating it (or not) with LightGBM, and explaining why, is exactly what a marketing-science team would expect.
- **Assumption check:** Gamma-Gamma assumes spend per order is independent of how often a customer orders. Check that correlation first and report it.
- **Where it fits:** a 90-day value leaderboard with baseline → BG/NBD+GG → Ridge/LightGBM, and a P(alive) vs calibrated P(churn) scatter in Customer 360.
- **Done when:** a test checks model fitting on a synthetic series generated from known BG/NBD parameters and recovers them within tolerance; leaderboard rows are logged to MLflow.

### P2 — Survival model for time-to-churn (optional, ~1.5 days)
- **What:** a discrete-time hazard model (logistic regression on customer-period rows) or Cox proportional hazards (`lifelines.CoxPHFitter`) on the same features. It gives a churn probability for **any** horizon (30/60/90/180 days), not just the fixed 90-day label.
- **Evaluation:** concordance index and integrated Brier score on the test cutoff. Also compare the model's 90-day probability with the calibrated classifier, using AUC and reliability curves.
- **Why:** it removes the arbitrariness of the fixed 90-day window and shows you know that censoring matters.

### P3 — Budget-constrained retention targeting (Tier A, ~1 day)
- **What:** turn scores into a *campaign decision*. Given budget B, contact cost c per channel, and an **assumed** save rate s (the share of contacted churners retained), choose which customers to contact to maximise `Σ p_churn · value · s − c`, subject to `Σ c ≤ B`. This is a knapsack problem; the greedy ratio rule is sufficient here.
- **Offline policy evaluation:** on the test cutoff, compare "value retained per £ spent" (using actual outcomes) across four policies:
  - (a) contact the top churn probabilities
  - (b) contact the top spenders
  - (c) the current rule table
  - (d) the optimised value-at-risk policy

  Run a **sensitivity grid over s ∈ {5%, 10%, 20%} and B**, because s is unknown without experiment data.
- **Honesty:** there is no treatment data, so the uplift (s) is an assumption. The report says so and proposes an A/B test design (sample size for detecting a 5-point save-rate lift) as the next step.
- **Where it fits:** a "Campaign Planner" dashboard page (budget slider → selected customers, expected retained value, CSV export), backed by an API endpoint `POST /campaign/plan`.
- **Done when:** unit tests confirm budget feasibility and that the chosen set is optimal on a small brute-force example; the policy comparison table is in the report.

### P4 — Product recommender (optional, ~1.5 days; PRD §1 and §2.3 mention personalised recommendations)
- **What:**
  - implicit-feedback ALS (`implicit`; fallback: item-item cosine on a `scipy.sparse` matrix)
  - compared against a popularity baseline and "bought-together" association rules
  - on a **time split** (train before the cutoff, evaluate on purchases in the following 90 days)
- **Metrics:** Recall@10, NDCG@10, catalogue coverage.
- **Where it fits:** "Recommended next products" in Customer 360 and `GET /customers/{id}/recommendations`. The recommendation engine (D5) can then attach "offer these items" to win-back actions.

### P5 — Production layer from the shared template (Tier A, ~2 days, reused by FlowCast and Northgate)
Build it here first:
- `uv` lock file
- `pandera` contracts for raw transactions and the processed feature table
- `dvc.yaml` pipeline, with `dvc repro` as the single command
- **Optuna** replacing the random search in Task 3 (time-series CV objective only, pruning, 50 trials for XGBoost/LightGBM)
- MLflow model registry (production model tagged "Production")
- **Evidently** drift report alongside the PSI table from D7
- pre-commit (ruff, mypy), GitHub Actions CI

Extract it as `../template/` for the other two projects.

### P6 — Grounded retention-message drafts with an LLM (optional, ~1 day)
- **What:** for a selected customer, an LLM (Claude API) drafts a short outreach message from a **facts JSON**: segment, top SHAP drivers, recommended action and products.
- **Guardrails:** a validator rejects any draft whose numbers or product names aren't in the facts JSON. Temperature 0. Prompt and version are logged.
- **Evaluation:** 30 customers scored on a small rubric (factually correct, tone, action matches the recommendation), reported as a table.
- **Why it's optional:** it needs an API key, and it's the least differentiating item. It's only worth doing *with* the grounding check and the evaluation. Without them it looks like a gimmick.
