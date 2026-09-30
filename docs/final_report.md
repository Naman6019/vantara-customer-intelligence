# VANTARA RETAIL SOLUTIONS
## Data Science & Analytics Division
### Product Final Technical Report — Customer Behavior Prediction Platform

---

## 1. Executive Summary & Business Context
Vantara Retail Solutions operates an omnichannel retail and e-commerce business serving direct consumers and small business resellers. Prior to this platform, marketing and retention teams relied on reactive, rule-based heuristics (e.g. "inactive for 90 days"), resulting in uniform discount distribution, late detection of disengaging high-value accounts, and suboptimal retention campaign ROI.

The **Customer Behavior Prediction Platform** solves these challenges by ingesting historical multi-year transaction logs to produce:
1. **Predictive Churn Risk Scoring**: Proactively identifying at-risk customers with **81.3% Recall** and **0.8080 Cross-Validated ROC-AUC**.
2. **Customer Lifetime Value (CLV) Forecasting**: Forecasting 90-day net spend with an **$R^2$ of 0.8447** and **MAE of £370.22**.
3. **Behavioral Customer Segmentation**: K-Means clustering identifying actionable personas: *High-Value Champions*, *Loyal Buyers*, *At-Risk High-Value*, and *Dormant Shoppers*.
4. **Deep Learning Benchmarks**: Feed-Forward ANN, Sequential LSTM on order trajectories, and an Unsupervised Autoencoder for anomaly detection.
5. **Model Explainability (XAI)**: Global and local SHAP, LIME explanations, Partial Dependence Plots (PDP), and plain-English narrative translations for marketing analysts.
6. **Production Serving**: Containerized FastAPI backend, PostgreSQL schema, and interactive Streamlit decision dashboard.

---

## 2. Data Engineering & Leak-Free Split Architecture
- **Source Data**: UCI Machine Learning Repository *Online Retail II* (`id=502`), comprising 1,067,371 records from 2009-12-01 to 2011-12-09.
- **Cleaning & Filtering**:
  - Dropped 243,007 rows with missing Customer ID for customer-level behavioral modeling.
  - Deduplicated 26,479 exact duplicate rows.
  - Filtered 3,374 administrative stock codes (`POST`, `D`, `M`, `BANK CHARGES`).
  - Preserved and flagged 17,610 returns and cancellations (`Quantity < 0` or Invoice starting with `'C'`) to compute customer return rates.
  - Cleaned dataset: **794,440 records** across **5,876 unique customers**, cached to high-performance Parquet format.
- **Strict Point-in-Time Split (Zero Target Leakage)**:
  - **Cutoff Date**: `2011-09-01 00:00:00`.
  - **Observation Window ($< T$)**: 623,150 transactions used strictly for feature engineering.
  - **Prediction Window ($\ge T$, 100 days)**: 171,290 transactions used strictly for target calculation.
  - Cohort: 5,225 active customers evaluated at cutoff. Churn rate in prediction window: **55.25%** (2,888 churned, 2,337 retained).

---

## 3. Supervised Model Leaderboards & Selection Justification

### 3.1 Classical Churn Classification Leaderboard (Held-Out Test Set: 784 Customers)

| Model | Test ROC-AUC | Test Recall (Churned Class) | Test Precision | Test F1-Score | Test Accuracy | 5-Fold CV ROC-AUC | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Production)** | **0.7846** | **81.29%** | 0.7111 | 0.7586 | 71.43% | **0.8080 ± 0.013** | 3.82s |
| **Random Forest** | 0.7903 | 79.91% | 0.7269 | 0.7613 | 72.32% | **0.8135 ± 0.016** | 2.18s |
| **Logistic Regression** | 0.7946 | 79.45% | 0.7560 | 0.7748 | 74.49% | **0.8079 ± 0.012** | 2.34s |
| **LightGBM** | 0.7904 | 77.83% | 0.7342 | 0.7556 | 72.19% | **0.8076 ± 0.015** | 2.30s |
| **Decision Tree** | 0.7720 | 80.37% | 0.7311 | 0.7657 | 72.83% | 0.7816 ± 0.016 | 1.54s |
| **KNN (k=7)** | 0.7548 | 77.83% | 0.7155 | 0.7456 | 70.66% | 0.7728 ± 0.014 | 1.58s |

**Selection Rationale**:
Per PRD Section 3.2 & 11, **Recall on the churned class is prioritized over raw accuracy** because the cost of failing to detect a churning high-value customer far outweighs a false alarm. **XGBoost** achieved the highest test recall (**81.29%**), surpassing the required target of $\ge 70\%$, with solid cross-validation stability (0.8080).

### 3.2 Customer Lifetime Value (CLV) Regression Leaderboard

| Model | Test $R^2$ Score | Test MAE (£) | Test RMSE (£) | Training Time |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression (Production)** | **0.8447** | **£370.22** | **£1,180.75** | **0.031s** |
| **Random Forest Regressor** | 0.6678 | £412.71 | £1,726.63 | 0.326s |
| **LightGBM Regressor** | 0.5793 | £449.50 | £1,943.20 | 0.053s |
| **XGBoost Regressor** | 0.5170 | £421.91 | £2,082.15 | 0.675s |
| **Gradient Boosting Regressor** | 0.3884 | £452.01 | £2,342.81 | 2.481s |

**Selection Rationale**:
**Ridge Regression** with $L_2$ regularization achieved an exceptional **$R^2$ of 0.8447**, comfortably exceeding the PRD requirement ($R^2 \ge 0.60$). Its linear penalty prevents overfitting on high-spend outliers while maintaining stable predictions.

---

## 4. Deep Learning Comparative Analysis

| Deep Learning Architecture | Objective / Input | Test Metric | Key Finding |
| :--- | :--- | :--- | :--- |
| **Feed-Forward ANN** | Tabular feature matrix with BatchNorm, Dropout(0.25), Early Stopping | ROC-AUC: **0.7895**<br>Recall: **85.22%** | Achieved the highest recall of all models (85.2%), confirming its utility as an ensemble consensus checker. |
| **Sequence LSTM** | Time-ordered order trajectories `[gap_days, spend, qty]` | ROC-AUC: **0.7740**<br>Recall: **80.54%** | Successfully learned non-static temporal cadence from sequential order intervals. |
| **Spending Autoencoder** | Unsupervised bottleneck reconstruction (`18 -> 12 -> 4 -> 12 -> 18`) | Reconstruction Threshold: **0.6736** (95th %ile) | Flagged **262 anomalous customer accounts** displaying irregular spending spikes or potential wholesale fraud. |

---

## 5. Customer Segmentation & Persona Profiles

Using empirical Elbow and Silhouette validation, $k=4$ was identified as the optimal cluster count:

| Cluster ID | Assigned Business Persona | Avg Recency | Avg Frequency | Avg Total Spend | Churn Rate | Strategic Action |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **0** | **At-Risk / Lapsed High-Value** | 224.5 days | 8.2 orders | £2,850.40 | 88.1% | **Immediate proactive outreach**, executive VIP concierge call. |
| **1** | **High-Value Champions (VIP)** | 28.3 days | 11.4 orders | £3,410.80 | 18.5% | Early access to new arrivals, loyalty tier rewards, no discounts needed. |
| **2** | **Dormant / Low-Engagement** | 268.1 days | 1.4 orders | £290.15 | 92.4% | Low-cost automated email re-engagement or discount incentives. |
| **3** | **Loyal & Active Buyers** | 42.1 days | 4.8 orders | £1,120.60 | 32.1% | Cross-sell and category-expansion recommendations. |

- **Clustering Validation**:
  - K-Means Silhouette Score: **0.4612**, Davies-Bouldin Index: **0.7565**.
  - GMM Silhouette Score: **0.1718**, confirming K-Means produces substantially tighter, more separable clusters.

---

## 6. Model Explainability (XAI)
1. **Global Macro Drivers**:
   - As visualized in `docs/shap_global_summary.png`, **Recency** and **Engagement Score** are the dominant predictors of churn, followed by **Purchase Frequency Trend** and **Total Items Bought**.
2. **Local Customer Explanations & Narratives**:
   - *Low-Risk Customer*: Churn Probability 1.8%. Mitigating factors: High engagement score (98.9, impact: -2.43) and minimal recency (0.4 days, impact: -0.79).
   - *High-Risk Customer*: Churn Probability 97.9%. Primary warning signs: Inactivity for 568 days (+0.86 impact) and collapsed engagement (+1.36 impact).
   - *Borderline Customer*: Churn Probability 50.0%. Mitigated by recent activity (29 days, -0.67 impact) but constrained by low basket volume (+0.19 impact).
3. **LIME vs SHAP Agreement**:
   - Both methods verified that recency $\le 48.6$ days and high engagement score reduce churn probability, confirming explanation consistency.

---

## 7. Mathematical Foundations Appendix

### 7.1 Logistic Regression & Binary Cross-Entropy Loss
For customer feature vector $\mathbf{x} \in \mathbb{R}^d$ and binary churn label $y \in \{0, 1\}$:
$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
The model parameters are optimized by minimizing Binary Cross-Entropy with $L_2$ penalty:
$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$

### 7.2 Ridge Regularization for CLV Regression
Future spend $y_{\text{clv}}$ is predicted using $L_2$-regularized least squares:
$$\min_{\mathbf{w}} \left\{ \frac{1}{2N} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \alpha \|\mathbf{w}\|_2^2 \right\}$$
The closed-form analytical solution is given by:
$$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

### 7.3 K-Means Clustering & Silhouette Coefficient
K-Means partitions $N$ customer observations into $k$ disjoint clusters $S = \{S_1, \dots, S_k\}$ minimizing within-cluster sum of squares (WCSS):
$$\arg\min_S \sum_{j=1}^k \sum_{\mathbf{x} \in S_j} \|\mathbf{x} - \boldsymbol{\mu}_j\|^2$$
The Silhouette Coefficient for observation $i$ is defined as:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
where $a(i)$ is mean intra-cluster distance and $b(i)$ is mean nearest-cluster distance.

### 7.4 Autoencoder Anomaly Reconstruction Loss
The spending autoencoder maps $\mathbf{x} \in \mathbb{R}^{18}$ to bottleneck $\mathbf{z} \in \mathbb{R}^4$ and reconstructs $\hat{\mathbf{x}}$:
$$\mathcal{L}_{\text{recon}} = \frac{1}{d} \sum_{j=1}^d (x_j - \hat{x}_j)^2$$
Customers whose $\mathcal{L}_{\text{recon}} > \tau_{95\%} = 0.6736$ are flagged for spending irregularity or bulk fraud review.

### 7.5 Classification Evaluation Metrics
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
$$\text{ROC-AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt$$

---

## 8. Deployment & Execution Verification
- **FastAPI Backend**: Operational on port `8000` with Swagger documentation (`/docs`), single-customer scoring (`/predict`), and batch CSV inference (`/batch-score`).
- **Streamlit Dashboard**: Operational on port `8501` featuring interactive 2D PCA segmentation, retention leaderboards, revenue trends, and dynamic XAI gauges.
- **Dockerization**: Runnable with a single command:
  ```bash
  docker-compose up --build
  ```
- **Test Suite**: 22 passing unit and integration tests across data validation, cleaning, leakage prevention, modeling, and API endpoints.

— *Vantara Retail Solutions Data Science Division* —
