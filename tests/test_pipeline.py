"""
tests/test_pipeline.py
========================
Unit and integration tests for the AccessGuard MLOps pipeline.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd

# ── Fixtures ───────────────────────────────────────────────────────────────────

SAMPLE_LOGINS = {
    "Username": [
        "alice",
        "alice",
        "bob",
        "bob",
        "carol",
        "carol",
        "alice",
        "bob",
        "carol",
        "alice",
        "bob",
    ],
    "IP": [
        "1.1.1.1",
        "1.1.1.1",
        "2.2.2.2",
        "3.3.3.3",
        "4.4.4.4",
        "4.4.4.4",
        "1.1.1.1",
        "3.3.3.3",
        "5.5.5.5",
        "9.9.9.9",
        "2.2.2.2",
    ],
    "Device": [
        "Windows 10",
        "Windows 10",
        "macOS",
        "macOS",
        "Ubuntu",
        "Ubuntu",
        "Windows 10",
        "macOS",
        "Ubuntu",
        "Windows 11",
        "macOS",
    ],
    "Browser": [
        "Chrome",
        "Chrome",
        "Firefox",
        "Firefox",
        "Edge",
        "Edge",
        "Chrome",
        "Firefox",
        "Edge",
        "Chrome",
        "Safari",
    ],
    "Timestamp": pd.date_range("2025-01-01", periods=11, freq="h").astype(str).tolist(),
    "MFA Enabled": [1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0],
    "Outcome": [0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1],
}

SAMPLE_USERS = {
    "User ID": [1000, 1001, 1002],
    "Username": ["alice", "bob", "carol"],
    "Password": ["hash1", "hash2", "hash3"],
    "MFA Enabled": [1, 0, 1],
    "IP": ["1.1.1.1", "2.2.2.2", "4.4.4.4"],
    "Device": ["Windows 10", "macOS", "Ubuntu"],
    "Browser": ["Chrome", "Firefox", "Edge"],
    "Registration Timestamp": [
        "2025-01-01T00:00:00",
        "2025-01-02T00:00:00",
        "2025-01-03T00:00:00",
    ],
}


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create temporary data directory with CSVs."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    pd.DataFrame(SAMPLE_LOGINS).to_csv(data_dir / "logins.csv", index=False)
    pd.DataFrame(SAMPLE_USERS).to_csv(data_dir / "users.csv", index=False)

    return tmp_path


# ── Tests: Data Loading ────────────────────────────────────────────────────────


class TestDataLoading:
    def test_load_valid_csv(self, tmp_data_dir, monkeypatch):
        from mlops.pipeline import train_pipeline as tp

        monkeypatch.setattr(tp, "LOGINS_CSV", tmp_data_dir / "data" / "logins.csv")
        monkeypatch.setattr(tp, "MODELS_DIR", tmp_data_dir / "models")
        monkeypatch.setattr(tp, "METRICS_DIR", tmp_data_dir / "mlops" / "monitoring")

        (tmp_data_dir / "models").mkdir(parents=True, exist_ok=True)
        (tmp_data_dir / "mlops" / "monitoring").mkdir(parents=True, exist_ok=True)

        df = tp.load_and_validate()
        assert len(df) == 11
        assert "Outcome" in df.columns

    def test_missing_csv_raises(self, tmp_path, monkeypatch):
        from mlops.pipeline import train_pipeline as tp

        monkeypatch.setattr(tp, "LOGINS_CSV", tmp_path / "none.csv")

        with pytest.raises(FileNotFoundError):
            tp.load_and_validate()


# ── Tests: Preprocessing ──────────────────────────────────────────────────────


class TestPreprocessing:
    def test_login_hour_created(self):
        from mlops.pipeline.train_pipeline import preprocess

        df = pd.DataFrame(SAMPLE_LOGINS)
        result = preprocess(df)

        assert "LoginHour" in result.columns
        assert result["LoginHour"].between(0, 23).all()

    def test_user_new_ip_created(self):
        from mlops.pipeline.train_pipeline import preprocess

        df = pd.DataFrame(SAMPLE_LOGINS)
        result = preprocess(df)

        assert "User_New_IP" in result.columns
        assert set(result["User_New_IP"].unique()).issubset({0, 1})


# ── Tests: Model Training ─────────────────────────────────────────────────────


class TestModelTraining:
    def test_train_returns_model(self):
        from mlops.pipeline.train_pipeline import preprocess, train

        df = pd.DataFrame(SAMPLE_LOGINS)
        df = preprocess(df)

        model, encoders, X_test, y_test, oenc = train(df)

        assert model is not None
        assert "IP" in encoders


# ── Tests: Evaluation ─────────────────────────────────────────────────────────


class TestEvaluation:
    def test_metrics_written(self, tmp_data_dir):
        from mlops.pipeline.train_pipeline import (
            preprocess,
            train,
            evaluate,
        )

        metrics_file = tmp_data_dir / "mlops" / "monitoring" / "metrics.json"
        metrics_file.parent.mkdir(parents=True)

        df = pd.DataFrame(SAMPLE_LOGINS)
        df = preprocess(df)

        model, encoders, X_test, y_test, oenc = train(df)

        import mlops.pipeline.train_pipeline as tp

        original = tp.METRICS_FILE
        tp.METRICS_FILE = metrics_file

        try:
            evaluate(model, X_test, y_test, oenc, encoders)
        finally:
            tp.METRICS_FILE = original

        assert metrics_file.exists()


# ── Tests: Drift ──────────────────────────────────────────────────────────────


class TestDrift:
    def test_no_drift(self):
        from mlops.monitoring.monitor_drift import check_drift

        metrics = {"accuracy": 0.9, "f1_weighted": 0.9}
        assert check_drift(metrics) == []


# ── Tests: End-to-End ─────────────────────────────────────────────────────────


class TestEndToEnd:
    def test_pipeline_runs(self, tmp_data_dir, monkeypatch):
        import mlops.pipeline.train_pipeline as tp

        monkeypatch.setattr(tp, "LOGINS_CSV", tmp_data_dir / "data" / "logins.csv")
        monkeypatch.setattr(tp, "MODEL_FILE", tmp_data_dir / "models" / "model.pkl")
        monkeypatch.setattr(
            tp,
            "METRICS_FILE",
            tmp_data_dir / "mlops" / "monitoring" / "metrics.json",
        )

        (tmp_data_dir / "models").mkdir(parents=True, exist_ok=True)
        (tmp_data_dir / "mlops" / "monitoring").mkdir(parents=True, exist_ok=True)

        exit_code = tp.run_pipeline()
        assert exit_code == 0
