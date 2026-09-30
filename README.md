# Vantara Retail Solutions — Customer Behavior Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

An end-to-end Machine Learning and Deep Learning platform for omnichannel retail customer analytics. The platform ingests multi-year transactional logs to predict customer churn, estimate customer lifetime value (CLV), model sequential purchase behavior, detect spending anomalies, and generate actionable customer segmentations with explainable AI (XAI).

---

## 🏗️ Repository Architecture

```
customer-behavior-prediction/
├── data/
│   ├── raw/                  # Original raw files and Parquet cache
│   ├── interim/              # Cleaned transactional records
│   └── processed/            # Final leak-free customer feature tables
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis & Hypotheses
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── src/
│   ├── data/                 # Ingestion, cleaning, and validation scripts
│   ├── features/             # RFM, dynamics, return rate, engagement scoring
│   ├── models/               # Classical ML (XGBoost, LightGBM, RF, etc.) & Deep Learning (ANN, LSTM, AE)
│   ├── segmentation/         # K-Means, GMM, clustering profiler
│   ├── explainability/       # SHAP and LIME wrappers & narrative translator
│   └── utils/                # Configuration, structured logging, helpers
├── api/                      # FastAPI service (single prediction, batch CSV scoring, metadata)
│   ├── main.py
│   ├── routers/
│   └── schemas/
├── frontend/                 # Interactive Streamlit dashboard
│   └── dashboard.py
├── models_artifacts/         # Serialized model weights, scalers, and encoders
├── config/
│   └── config.yaml           # Centralized configuration & hyperparameters
├── tests/                    # Pytest test suite (including strict target leakage checks)
├── docs/                     # System architecture & database ER diagrams
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Environment Setup

```bash
# Clone and enter directory
cd Internmo

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download and Load Dataset

The dataset used is the authoritative **UCI Online Retail II** dataset (`id=502`), containing ~1.06M transaction line items across 2009–2011.

```bash
# Automatically downloads from UCI repository, standardizes schema, and caches to Parquet
python -m src.data.loader
```

### 3. Run Automated Validation & Tests

```bash
pytest tests/ -v --cov=src
```

---

## 📊 Core Analytical Pipeline

1. **Temporal Cutoff & Leakage Prevention**: Features are computed strictly using transactions prior to cutoff $T$; churn (90-day inactivity) and CLV are computed in window $[T, T + 90\text{ days}]$.
2. **Models**:
   - **Churn Classification**: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, KNN/SVM, and Feed-Forward ANN.
   - **CLV Regression**: Ridge, Random Forest, and LightGBM Regressors.
   - **Sequence Modeling**: LSTM on customer purchase trajectories.
   - **Anomaly Detection**: Unsupervised Autoencoder on spending patterns.
3. **Customer Segmentation**: K-Means + Gaussian Mixture Models mapped to business personas.
4. **Explainability (XAI)**: Global and local SHAP, LIME, Partial Dependence Plots, and plain-language narrative translation.

---

## 📄 License & Attribution
- Dataset: Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. Distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
