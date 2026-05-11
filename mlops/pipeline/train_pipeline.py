"""
AccessGuard MLOps Training Pipeline
====================================
Triggered automatically when:
  - A new user registers (users.csv updated)
  - GitHub Actions runs on push/PR
  - Manual trigger via scripts/retrain.py

Steps:
  1. Load & validate data
  2. Preprocess features
  3. Train RandomForest model
  4. Evaluate & log metrics
  5. Save model artifact + versioned copy
  6. Write metrics to mlops/monitoring/metrics.json
"""

import json
import hashlib
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "mlops" / "monitoring"

LOGINS_CSV = DATA_DIR / "logins.csv"
USERS_CSV = DATA_DIR / "users.csv"
MODEL_FILE = MODELS_DIR / "accessguard_model.pkl"
METRICS_FILE = METRICS_DIR / "metrics.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(METRICS_DIR / "pipeline.log"),
    ],
)
log = logging.getLogger("train_pipeline")


# ── Step 1: Load & Validate Data ───────────────────────────────────────────────
def load_and_validate():
    log.info("STEP 1 — Loading data from %s", LOGINS_CSV)
    if not LOGINS_CSV.exists():
        raise FileNotFoundError(f"logins.csv not found at {LOGINS_CSV}")

    df = pd.read_csv(LOGINS_CSV)
    log.info("Loaded %d rows, columns: %s", len(df), list(df.columns))

    required_cols = [
        "Username",
        "IP",
        "Device",
        "Browser",
        "Timestamp",
        "MFA Enabled",
        "Outcome",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    initial_rows = len(df)
    df = df.dropna(subset=["Outcome"])
    df = df.dropna(subset=["IP", "Device", "Browser", "MFA Enabled", "Username"])
    log.info("After cleaning: %d / %d rows retained", len(df), initial_rows)

    if len(df) < 5:
        raise ValueError(
            f"Insufficient training data: only {len(df)} rows after cleaning. "
            "Need at least 5. Register more users and log more logins first."
        )
    return df


# ── Step 2: Preprocess ─────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    log.info("STEP 2 — Preprocessing features")

    # Clean MFA Enabled
    df["MFA Enabled"] = pd.to_numeric(df["MFA Enabled"], errors="coerce")
    df = df[df["MFA Enabled"].notna()].copy()
    df["MFA Enabled"] = df["MFA Enabled"].astype(int)

    # Login Hour feature
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["LoginHour"] = df["Timestamp"].dt.hour.fillna(0).astype(int)

    # User-specific new IP feature (chronological)
    known_ips: dict = {}
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["User_New_IP"] = 0
    for idx, row in df.iterrows():
        user, ip = row["Username"], row["IP"]
        known_ips.setdefault(user, set())
        if ip not in known_ips[user]:
            df.at[idx, "User_New_IP"] = 1
        known_ips[user].add(ip)

    log.info("Features engineered: LoginHour, User_New_IP")
    return df


# ── Step 3: Train ──────────────────────────────────────────────────────────────
def train(df: pd.DataFrame):
    log.info("STEP 3 — Training RandomForestClassifier")

    features = ["IP", "Device", "Browser", "MFA Enabled", "LoginHour", "User_New_IP"]
    X = df[features].copy().astype(str)

    encoders: dict = {}
    for col in ["IP", "Device", "Browser"]:
        enc = LabelEncoder()
        X[col] = enc.fit_transform(X[col])
        encoders[col] = enc

    X["MFA Enabled"] = df["MFA Enabled"].astype(int)
    X["LoginHour"] = df["LoginHour"].astype(int)
    X["User_New_IP"] = df["User_New_IP"].astype(int)

    y_str = df["Outcome"].astype(str).fillna("Unknown")
    outcome_enc = LabelEncoder()
    y = outcome_enc.fit_transform(y_str)
    encoders["Outcome"] = outcome_enc

    # Guard against single-class targets
    unique_classes = np.unique(y)
    stratify = y if len(unique_classes) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    log.info("Model trained on %d samples", len(X_train))
    return model, encoders, X_test, y_test, outcome_enc


# ── Step 4: Evaluate & Log Metrics ────────────────────────────────────────────
def evaluate(model, X_test, y_test, outcome_enc, encoders):
    log.info("STEP 4 — Evaluating model")

    y_pred = model.predict(X_test)
    labels = outcome_enc.classes_

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=labels, zero_division=0)

    # AUC (only if binary classification)
    auc = None
    if len(labels) == 2:
        proba = model.predict_proba(X_test)[:, 1]
        auc = round(roc_auc_score(y_test, proba), 4)

    log.info("Accuracy: %.4f | F1: %.4f | AUC: %s", acc, f1, auc)
    log.info("\n%s", report)

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classes": list(labels),
        "train_samples": int(model.n_features_in_),
        "n_estimators": model.n_estimators,
    }

    # Append to metrics history
    history = []
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as fh:
            try:
                history = json.load(fh)
                if isinstance(history, dict):
                    history = [history]
            except json.JSONDecodeError:
                history = []
    history.append(metrics)
    with open(METRICS_FILE, "w") as fh:
        json.dump(history, fh, indent=2)
    log.info("Metrics saved to %s", METRICS_FILE)
    return metrics


# ── Step 5: Save Model ─────────────────────────────────────────────────────────
def save_model(model, encoders):
    log.info("STEP 5 — Saving model artifact")

    joblib.dump((model, encoders), MODEL_FILE)
    log.info("Model saved -> %s", MODEL_FILE)

    # Versioned copy: models/accessguard_model_<timestamp>.pkl
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ver = MODELS_DIR / f"accessguard_model_{ts}.pkl"
    joblib.dump((model, encoders), ver)
    log.info("Versioned copy -> %s", ver)

    # Checksum for integrity
    with open(MODEL_FILE, "rb") as fh:
        sha256 = hashlib.sha256(fh.read()).hexdigest()
    checksum_file = MODEL_FILE.with_suffix(".sha256")
    checksum_file.write_text(sha256)
    log.info("SHA-256 checksum: %s", sha256)
    return str(ver)


# ── Main Entrypoint ────────────────────────────────────────────────────────────
def run_pipeline():
    log.info("=" * 60)
    log.info("AccessGuard MLOps Training Pipeline — START")
    log.info("=" * 60)

    try:
        df = load_and_validate()
        df = preprocess(df)
        model, encoders, Xt, yt, oenc = train(df)
        metrics = evaluate(model, Xt, yt, oenc, encoders)
        versioned_path = save_model(model, encoders)

        log.info("=" * 60)
        log.info("Pipeline completed successfully")
        log.info("  Accuracy : %.4f", metrics["accuracy"])
        log.info("  F1 Score : %.4f", metrics["f1_weighted"])
        log.info("  Model    : %s", versioned_path)
        log.info("=" * 60)
        return 0

    except Exception as exc:
        log.exception("Pipeline FAILED: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(run_pipeline())
