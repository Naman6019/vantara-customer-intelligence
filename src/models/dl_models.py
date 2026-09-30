from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger
from src.models.data_split import split_data

logger = get_logger("deep_learning")


# ==========================================
# 1. Feed-Forward ANN (PRD Section 8.2)
# ==========================================
class ChurnANN(nn.Module):
    """
    Feed-forward ANN with Batch Normalization, Dropout, and non-linear activations
    as specified in PRD Section 8.2.
    """
    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout_rate: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_churn_ann(
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 0.001,
    patience: int = 7,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Trains and validates the feed-forward ANN classifier with early stopping.
    """
    logger.info("Preparing data tensors for ANN training...")
    data = split_data()

    X_train = torch.tensor(data["X_train_scaled"].values, dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(data["X_val_scaled"].values, dtype=torch.float32)
    y_val = torch.tensor(data["y_val"], dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(data["X_test_scaled"].values, dtype=torch.float32)
    y_test = data["y_test"]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = ChurnANN(input_dim=X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                val_loss += criterion(logits, batch_y).item() * len(batch_x)
        val_loss /= len(X_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Test set evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test).squeeze()
        test_probs = torch.sigmoid(test_logits).numpy()
        test_preds = (test_probs >= 0.5).astype(int)

    metrics = {
        "Model": "Neural Network (ANN)",
        "Test ROC-AUC": round(float(roc_auc_score(y_test, test_probs)), 4),
        "Test Recall": round(float(recall_score(y_test, test_preds)), 4),
        "Test Precision": round(float(precision_score(y_test, test_preds)), 4),
        "Test F1": round(float(f1_score(y_test, test_preds)), 4),
        "Test Accuracy": round(float(accuracy_score(y_test, test_preds)), 4),
    }
    logger.info(f"ANN Test Evaluation: ROC-AUC={metrics['Test ROC-AUC']}, Recall={metrics['Test Recall']}")

    if save_artifacts:
        root = get_project_root()
        art_dir = root / "models_artifacts"
        torch.save(model.state_dict(), art_dir / "ann_churn_model.pt")
        logger.info("Saved ANN model weights to models_artifacts/ann_churn_model.pt")

    return {"model": model, "metrics": metrics}


# ==========================================
# 2. Unsupervised Autoencoder (PRD Section 8.2)
# ==========================================
class SpendingAutoencoder(nn.Module):
    """
    Unsupervised Autoencoder trained on customer spending profiles.
    High reconstruction error flags anomalies, potential fraud, or extreme whale outliers.
    """
    def __init__(self, input_dim: int, latent_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 12),
            nn.ReLU(),
            nn.Linear(12, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)


def train_spending_autoencoder(epochs: int = 35, batch_size: int = 64, save_artifacts: bool = True) -> Dict[str, Any]:
    """
    Trains the Autoencoder and determines the anomaly reconstruction threshold (95th percentile).
    """
    logger.info("Training Unsupervised Spending Autoencoder...")
    data = split_data()
    X_train = torch.tensor(data["X_train_scaled"].values, dtype=torch.float32)
    X_all = torch.tensor(np.vstack([
        data["X_train_scaled"].values,
        data["X_val_scaled"].values,
        data["X_test_scaled"].values
    ]), dtype=torch.float32)

    model = SpendingAutoencoder(input_dim=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for (batch_x,) in loader:
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()

    # Calculate reconstruction errors across all customers
    model.eval()
    with torch.no_grad():
        reconstructed_all = model(X_all)
        mse_errors = torch.mean((X_all - reconstructed_all) ** 2, dim=1).numpy()

    threshold = float(np.percentile(mse_errors, 95))
    anomaly_count = int((mse_errors > threshold).sum())
    logger.info(f"Autoencoder 95th percentile reconstruction threshold: {threshold:.4f} ({anomaly_count} anomalous accounts)")

    if save_artifacts:
        root = get_project_root()
        art_dir = root / "models_artifacts"
        torch.save(model.state_dict(), art_dir / "autoencoder_model.pt")
        joblib.dump(threshold, art_dir / "autoencoder_threshold.joblib")
        logger.info("Saved Autoencoder model and anomaly threshold artifact.")

    return {"model": model, "threshold": threshold, "errors": mse_errors}


# ==========================================
# 3. Sequence LSTM Model (PRD Section 8.2)
# ==========================================
class PurchaseLSTM(nn.Module):
    """
    LSTM sequence classifier trained on ordered sequences of customer order gaps and spend.
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (hn, _) = self.lstm(x)
        last_hidden = hn[-1]
        return self.fc(last_hidden)


def prepare_lstm_sequences(max_seq_len: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts time-ordered purchase sequences per customer (gap_days, spend, quantity).
    Pads sequences up to max_seq_len.
    """
    root = get_project_root()
    df_cleaned = pd.read_parquet(root / "data/interim/cleaned_transactions.parquet")
    config = load_config()
    cutoff = pd.Timestamp(config.get("features", {}).get("temporal", {}).get("cutoff_date", "2011-09-01"))
    
    # Observation records
    obs_df = df_cleaned[(df_cleaned["invoice_date"] < cutoff) & (df_cleaned["quantity"] > 0) & (~df_cleaned["is_cancellation"])].copy()
    
    # Customer order summaries
    orders = (
        obs_df.groupby(["customer_id", "invoice"])
        .agg(order_date=("invoice_date", "min"), spend=("line_total", "sum"), qty=("quantity", "sum"))
        .reset_index()
        .sort_values(by=["customer_id", "order_date"])
    )
    orders["prev_date"] = orders.groupby("customer_id")["order_date"].shift(1)
    orders["gap_days"] = (orders["order_date"] - orders["prev_date"]).dt.total_seconds().fillna(0) / 86400.0

    # Targets
    pred_df = df_cleaned[(df_cleaned["invoice_date"] >= cutoff) & (df_cleaned["quantity"] > 0) & (~df_cleaned["is_cancellation"])]
    active_pred_customers = set(pred_df["customer_id"].unique())

    sequences = []
    labels = []
    
    for cust_id, group in orders.groupby("customer_id"):
        # Features: [gap_days / 100, log1p(spend), log1p(qty)]
        feats = np.column_stack([
            group["gap_days"].values / 100.0,
            np.log1p(np.maximum(0, group["spend"].values)),
            np.log1p(np.maximum(0, group["qty"].values))
        ])
        
        # Pad or slice to max_seq_len
        if len(feats) >= max_seq_len:
            seq = feats[-max_seq_len:]
        else:
            pad = np.zeros((max_seq_len - len(feats), 3))
            seq = np.vstack([pad, feats])
            
        is_churned = 0 if cust_id in active_pred_customers else 1
        sequences.append(seq)
        labels.append(is_churned)

    return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(1)


def train_purchase_lstm(epochs: int = 25, batch_size: int = 64, save_artifacts: bool = True) -> Dict[str, Any]:
    """
    Trains the sequence LSTM model to predict churn likelihood from purchase timing sequences.
    """
    logger.info("Preparing purchase order sequences for LSTM training...")
    X_seq, y_seq = prepare_lstm_sequences()

    # 80/20 train/test split
    n = len(X_seq)
    train_size = int(0.8 * n)
    indices = np.random.RandomState(42).permutation(n)
    tr_idx, te_idx = indices[:train_size], indices[train_size:]

    train_loader = DataLoader(TensorDataset(X_seq[tr_idx], y_seq[tr_idx]), batch_size=batch_size, shuffle=True)
    X_test, y_test = X_seq[te_idx], y_seq[te_idx].numpy().squeeze()

    model = PurchaseLSTM(input_dim=3, hidden_dim=32)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test).squeeze()
        test_probs = torch.sigmoid(test_logits).numpy()
        test_preds = (test_probs >= 0.5).astype(int)

    auc = round(float(roc_auc_score(y_test, test_probs)), 4)
    rec = round(float(recall_score(y_test, test_preds)), 4)
    logger.info(f"Sequence LSTM Evaluation: Test ROC-AUC={auc}, Recall={rec}")

    if save_artifacts:
        root = get_project_root()
        art_dir = root / "models_artifacts"
        torch.save(model.state_dict(), art_dir / "lstm_churn_model.pt")
        logger.info("Saved LSTM model weights to models_artifacts/lstm_churn_model.pt")

    return {
        "Model": "Sequence LSTM",
        "Test ROC-AUC": auc,
        "Test Recall": rec,
        "Test Precision": round(float(precision_score(y_test, test_preds)), 4),
        "Test F1": round(float(f1_score(y_test, test_preds)), 4),
        "Test Accuracy": round(float(accuracy_score(y_test, test_preds)), 4),
    }


if __name__ == "__main__":
    ann_res = train_churn_ann()
    print("\n--- ANN Model Metrics ---")
    print(ann_res["metrics"])

    lstm_res = train_purchase_lstm()
    print("\n--- LSTM Sequence Model Metrics ---")
    print(lstm_res)

    ae_res = train_spending_autoencoder()
    print(f"\n--- Autoencoder Anomaly Threshold (95th %ile): {ae_res['threshold']:.4f} ---")
