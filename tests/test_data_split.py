import joblib
import pytest

from src.models import data_split
from src.models.data_split import (
    FEATURE_COLS,
    load_processed_feature_table,
    split_data,
)


def test_split_data_partitions_sum_to_total(feature_table):
    data = split_data(feature_table, save_scaler=False)
    total = len(data["X_train"]) + len(data["X_val"]) + len(data["X_test"])
    assert total == len(feature_table)
    assert data["X_train"].shape[1] == len(FEATURE_COLS)


def test_split_data_scaled_columns_match(feature_table):
    data = split_data(feature_table, save_scaler=False)
    for key in ["X_train", "X_val", "X_test"]:
        assert list(data[key].columns) == FEATURE_COLS
        scaled = data[key + "_scaled"]
        assert list(scaled.columns) == FEATURE_COLS
        assert scaled.shape == data[key].shape
    # Scaler fitted on train only -> train mean ~ 0
    assert abs(data["X_train_scaled"].mean().mean()) < 0.2


def test_split_data_saves_scaler(feature_table, monkeypatch, tmp_path):
    monkeypatch.setattr(data_split, "get_project_root", lambda: tmp_path)
    split_data(feature_table, save_scaler=True)
    scaler_path = tmp_path / "models_artifacts/scaler.joblib"
    assert scaler_path.exists()
    scaler = joblib.load(scaler_path)
    assert hasattr(scaler, "transform")


def test_load_processed_feature_table_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(data_split, "get_project_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="Processed feature table not found"):
        load_processed_feature_table()


def test_load_processed_feature_table_reads_parquet(feature_table, monkeypatch, tmp_path):
    monkeypatch.setattr(data_split, "get_project_root", lambda: tmp_path)
    proc = tmp_path / "data/processed"
    proc.mkdir(parents=True)
    feature_table.to_parquet(proc / "customer_features.parquet")

    loaded = load_processed_feature_table()
    assert len(loaded) == len(feature_table)
