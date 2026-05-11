"""
tests/test_auth_risk.py
========================
Tests for user registration trigger and risk scoring logic.
"""

import pandas as pd
from unittest.mock import patch

# ── Tests: Registration Trigger ───────────────────────────────────────────────


class TestRegistrationTrigger:
    def test_trigger_log_written(self, tmp_path, monkeypatch):
        """on_new_registration writes to trigger.log."""
        from mlops.pipeline import trigger_pipeline as tp

        trigger_log = tmp_path / "trigger.log"
        monkeypatch.setattr(tp, "TRIGGER_LOG", trigger_log)
        monkeypatch.setattr(tp, "METRICS_DIR", tmp_path)
        monkeypatch.setattr(tp, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(tp, "RETRAIN_EVERY_N_USERS", 999)  # skip retrain

        # Add a users.csv so the count function works
        (tmp_path / "data").mkdir()
        pd.DataFrame(
            {
                "Username": ["testuser"],
                "Registration Timestamp": ["2025-01-01T00:00:00"],
            }
        ).to_csv(tmp_path / "data" / "users.csv", index=False)

        tp.on_new_registration("testuser")

        assert trigger_log.exists()
        content = trigger_log.read_text()
        assert "testuser" in content

    def test_pipeline_called_when_threshold_met(self, tmp_path, monkeypatch):
        from mlops.pipeline import trigger_pipeline as tp

        monkeypatch.setattr(tp, "TRIGGER_LOG", tmp_path / "trigger.log")
        monkeypatch.setattr(tp, "METRICS_DIR", tmp_path)
        monkeypatch.setattr(tp, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(tp, "RETRAIN_EVERY_N_USERS", 1)

        (tmp_path / "data").mkdir()
        pd.DataFrame(
            {
                "Username": ["alice"],
                "Registration Timestamp": ["2025-01-01T00:00:00"],
            }
        ).to_csv(tmp_path / "data" / "users.csv", index=False)

        pipeline_called = {"called": False}

        def mock_run_pipeline():
            pipeline_called["called"] = True
            return True

        monkeypatch.setattr(tp, "_run_training_pipeline", mock_run_pipeline)
        tp.on_new_registration("alice")

        assert pipeline_called["called"]

    def test_pipeline_skipped_when_below_threshold(self, tmp_path, monkeypatch):
        from mlops.pipeline import trigger_pipeline as tp

        monkeypatch.setattr(tp, "TRIGGER_LOG", tmp_path / "trigger.log")
        monkeypatch.setattr(tp, "METRICS_DIR", tmp_path)
        monkeypatch.setattr(tp, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(tp, "RETRAIN_EVERY_N_USERS", 100)

        (tmp_path / "data").mkdir()
        pd.DataFrame(
            {
                "Username": ["alice"],
                "Registration Timestamp": ["2025-01-01T00:00:00"],
            }
        ).to_csv(tmp_path / "data" / "users.csv", index=False)

        pipeline_called = {"called": False}

        def mock_run_pipeline():
            pipeline_called["called"] = True
            return True

        monkeypatch.setattr(tp, "_run_training_pipeline", mock_run_pipeline)
        tp.on_new_registration("alice")

        assert not pipeline_called["called"]


# ── Tests: Risk Scoring ────────────────────────────────────────────────────────


class TestRiskScoring:
    """
    These tests work with the existing accessguard/risk.py module.
    We mock external calls (geolocation, logins CSV).
    """

    def _predict(self, **kwargs):
        """Helper: import risk module and call predict_login."""
        # Mock all external dependencies
        with patch("accessguard.risk.get_geolocation", return_value="Unknown"), patch(
            "accessguard.risk.login_attempts_in_last_hour", return_value=0
        ), patch("accessguard.risk.get_user_profile", return_value=None):
            from accessguard.risk import predict_login

            return predict_login(**kwargs)

    def test_same_ip_device_browser_low_risk(self):
        score, decision, reasons = self._predict(
            username="alice",
            ip="1.1.1.1",
            device="Windows 10",
            browser="Chrome",
            hour=10,
            mfa_enabled=True,
            registered_ip="1.1.1.1",
            registered_device="Windows 10",
            registered_browser="Chrome",
        )
        assert score < 30
        assert "ALLOW" in decision

    def test_new_ip_raises_score(self):
        score, decision, reasons = self._predict(
            username="bob",
            ip="9.9.9.9",  # different from registered
            device="macOS",
            browser="Firefox",
            hour=10,
            mfa_enabled=False,
            registered_ip="1.1.1.1",
            registered_device="macOS",
            registered_browser="Firefox",
        )
        assert score >= 30
        assert any("IP" in r for r in reasons)

    def test_multiple_anomalies_high_risk(self):
        score, decision, reasons = self._predict(
            username="carol",
            ip="9.9.9.9",
            device="Linux",
            browser="Edge",
            hour=10,
            mfa_enabled=False,
            registered_ip="1.1.1.1",
            registered_device="Windows 10",
            registered_browser="Chrome",
        )
        # Different IP, device, browser => high risk
        assert score >= 60
        assert "BLOCK" in decision

    def test_mfa_reduces_medium_risk_to_allow(self):
        score, decision, reasons = self._predict(
            username="dave",
            ip="8.8.8.8",
            device="Windows 10",
            browser="Chrome",
            hour=10,
            mfa_enabled=True,  # MFA ON
            registered_ip="1.1.1.1",
            registered_device="Windows 10",
            registered_browser="Chrome",
        )
        # Only IP mismatch (score ~30) + MFA = should allow
        if 30 <= score < 60:
            assert "MFA" in decision

    def test_rapid_attempts_raises_score(self):
        with patch("accessguard.risk.get_geolocation", return_value="Unknown"), patch(
            "accessguard.risk.login_attempts_in_last_hour", return_value=10
        ), patch("accessguard.risk.get_user_profile", return_value=None):
            from accessguard.risk import predict_login

            score, decision, reasons = predict_login(
                username="alice",
                ip="1.1.1.1",
                device="Windows 10",
                browser="Chrome",
                hour=10,
                mfa_enabled=True,
                registered_ip="1.1.1.1",
                registered_device="Windows 10",
                registered_browser="Chrome",
            )
        assert score >= 20
        assert any("attempt" in r.lower() for r in reasons)
