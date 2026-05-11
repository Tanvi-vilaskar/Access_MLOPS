# AccessGuard — MLOps Pipeline & CI/CD Guide

> **What this project does:** AccessGuard is an AI-powered login security system that uses a Random Forest model to score every login attempt for risk. When a new user registers, the system automatically retrains the model so it stays current with your user base.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [How the Pipeline Works — Overview](#2-how-the-pipeline-works--overview)
3. [Step-by-Step: From Registration to Deployed Model](#3-step-by-step-from-registration-to-deployed-model)
4. [GitHub Actions CI/CD — Every Job Explained](#4-github-actions-cicd--every-job-explained)
5. [Running Locally](#5-running-locally)
6. [GitHub Secrets to Configure](#6-github-secrets-to-configure)
7. [Monitoring & Drift Detection](#7-monitoring--drift-detection)
8. [Extending the Pipeline](#8-extending-the-pipeline)

---

## 1. Project Structure

```
AccessGuard_MLOps/
│
├── .github/
│   └── workflows/
│       └── mlops_pipeline.yml      ← CI/CD pipeline (5 jobs)
│
├── accessguard/                    ← Main application code
│   ├── auth.py                     ← Registration + MLOps trigger hook  ⬅ key file
│   ├── model.py                    ← RandomForest training (original)
│   ├── risk.py                     ← Login risk scoring engine
│   ├── data_handler.py             ← CSV read/write helpers
│   ├── config.py                   ← File paths
│   ├── utils.py                    ← IP/device/geo helpers
│   ├── alerts.py                   ← Slack/log alerting
│   ├── main.py                     ← Streamlit app entry point
│   └── ...                         ← (other modules)
│
├── mlops/
│   ├── pipeline/
│   │   ├── train_pipeline.py       ← Full 5-step MLOps training pipeline ⬅ key file
│   │   └── trigger_pipeline.py     ← Hook called on new registration     ⬅ key file
│   └── monitoring/
│       ├── monitor_drift.py        ← Drift detection script              ⬅ key file
│       ├── metrics.json            ← Metrics history (auto-generated)
│       ├── pipeline.log            ← Training run log (auto-generated)
│       └── trigger.log             ← Registration event log (auto-generated)
│
├── models/
│   ├── accessguard_model.pkl       ← Latest trained model artifact
│   ├── accessguard_model.sha256    ← Integrity checksum
│   └── accessguard_model_<ts>.pkl  ← Versioned backups (auto-generated)
│
├── data/
│   ├── logins.csv                  ← Login event log (grows over time)
│   └── users.csv                   ← Registered users
│
├── tests/
│   ├── test_pipeline.py            ← Pipeline unit + integration tests
│   └── test_auth_risk.py           ← Auth registration + risk scoring tests
│
├── scripts/
│   └── retrain.py                  ← Manual retrain utility
│
├── requirements.txt
├── setup.cfg                       ← pytest + coverage config
└── .gitignore
```

---

## 2. How the Pipeline Works — Overview

```
User fills Registration form (Streamlit)
         │
         ▼
  auth.register_user()
  ├── Validate username unique
  ├── Capture face video (if camera available)
  ├── Hash password (bcrypt)
  ├── Append row to data/users.csv
  └── ► TRIGGER: on_new_registration(username)
                    │
                    ▼
         trigger_pipeline.py
         ├── Log event → trigger.log
         ├── Count new registrations since last train
         └── If count ≥ threshold (default: 1):
                    │
                    ▼
         train_pipeline.run_pipeline()
         ├── STEP 1: Load & validate logins.csv
         ├── STEP 2: Feature engineering (LoginHour, User_New_IP)
         ├── STEP 3: Train RandomForestClassifier
         ├── STEP 4: Evaluate → accuracy, F1, AUC → metrics.json
         └── STEP 5: Save models/accessguard_model.pkl + versioned copy
```

In **GitHub Actions**, the same pipeline runs automatically when data files or code changes are pushed.

---

## 3. Step-by-Step: From Registration to Deployed Model

### Step 1 — User Registers in the App

The user fills in username, password, and MFA preference in the Streamlit UI (`accessguard/main.py`). When they click **Register**, `auth.register_user()` runs:

- Checks the username is not already in `data/users.csv`
- Records system info: IP address, device (OS), browser
- Hashes the password with bcrypt (never stored in plain text)
- Appends a new row to `data/users.csv`
- **Calls `on_new_registration(username)`** from `mlops/pipeline/trigger_pipeline.py`

### Step 2 — Trigger Decides Whether to Retrain

`trigger_pipeline.py:on_new_registration()`:

1. Writes a JSON event to `mlops/monitoring/trigger.log`
2. Reads `data/users.csv` and checks `Registration Timestamp` against the last training timestamp in `metrics.json`
3. If `new_registrations >= RETRAIN_EVERY_N_USERS` (env var, default `1`): runs `train_pipeline.run_pipeline()`
4. If the env var `ENABLE_GIT_PUSH=true` is set, commits and pushes the updated model to the repo

To change the threshold (e.g. retrain every 5 new users):
```bash
export RETRAIN_EVERY_N_USERS=5
```

### Step 3 — Training Pipeline Runs (5 Sub-steps)

File: `mlops/pipeline/train_pipeline.py`

| Sub-step | What happens |
|----------|-------------|
| **1. Load & Validate** | Reads `data/logins.csv`, checks required columns exist, drops null rows. Raises `ValueError` if fewer than 5 usable rows remain. |
| **2. Preprocess** | Parses timestamps → extracts `LoginHour` (0–23). Builds `User_New_IP` flag: walks logins in chronological order per user; flags 1 if the IP has not been seen before. |
| **3. Train** | Label-encodes IP, Device, Browser. Splits 80/20 train/test. Trains `RandomForestClassifier(n_estimators=100, class_weight='balanced')`. |
| **4. Evaluate** | Computes accuracy, weighted F1, ROC-AUC. Prints confusion matrix. Appends metrics dict to `mlops/monitoring/metrics.json`. |
| **5. Save** | Saves `(model, encoders)` tuple via `joblib` to `models/accessguard_model.pkl`. Creates a timestamped versioned copy. Writes SHA-256 checksum. |

### Step 4 — Risk Scoring Uses the New Model

At login time, `accessguard/risk.py:predict_login()` loads the saved `accessguard_model.pkl` (via `model.py:load_model()`) and scores the attempt. Because the model was just retrained with the latest data, it now knows about the new user's device, IP, and browser patterns.

### Step 5 — CI/CD Pipeline Validates Everything

When you push code or data to GitHub, the Actions workflow (`.github/workflows/mlops_pipeline.yml`) runs all 5 jobs (see next section). On success on `main`, it commits the updated model artifact back to the repository.

---

## 4. GitHub Actions CI/CD — Every Job Explained

### Triggers

| Event | What runs |
|-------|-----------|
| Push to `main` or `develop` (data/code changed) | All 5 jobs |
| Pull Request to `main` | Jobs 1 (lint) + 2 (test) only |
| Manual `workflow_dispatch` | All 5 jobs (retrain forced) |
| Nightly `schedule` (02:00 UTC) | All 5 jobs (drift check + retrain) |

### Job 1 — Lint & Format Check

```
Runs: black --check, flake8
Why: Catch syntax errors and style violations before any code runs.
Fails if: Any file in accessguard/, mlops/, tests/ has format or style issues.
Fix: Run `black .` locally before pushing.
```

### Job 2 — Run Tests

```
Runs: pytest tests/ --cov=accessguard --cov=mlops
Why: Verify data loading, preprocessing, training, evaluation, and drift logic.
Fails if: Any test fails OR coverage drops below 60% (set in setup.cfg).
Output artifact: coverage.xml (downloadable from Actions run page).
```

Tests cover:
- `test_pipeline.py` — data validation, preprocessing correctness, model training, metric writing, full end-to-end run
- `test_auth_risk.py` — registration trigger logic, risk scoring (same IP = low score, new IP = higher score, multiple anomalies = block, rapid attempts = flagged)

### Job 3 — Train Model

```
Runs: python mlops/pipeline/train_pipeline.py
Why: Produce an up-to-date model artifact from the latest CSV data.
Condition: Only runs when logins.csv / users.csv / model code changed, or on manual/scheduled trigger.
Output artifacts: models/accessguard_model.pkl, metrics.json, pipeline.log
```

This job uses the real data in your repository. If you commit an updated `data/logins.csv`, this job automatically retrains.

### Job 4 — Drift Detection

```
Runs: python mlops/monitoring/monitor_drift.py
Why: Prevent a badly-performing model from being deployed.
Fails if:
  - accuracy < 0.70
  - f1_weighted < 0.65
On failure: Sends a Slack alert (if SLACK_WEBHOOK_URL secret is configured).
Output artifact: drift_report.txt
```

To change thresholds, edit `THRESHOLDS` dict in `mlops/monitoring/monitor_drift.py`.

### Job 5 — Deploy Model Artifact

```
Runs: git commit + git push (model files only)
Condition: Only on main branch, after drift check passes.
Why: Keeps the repository's model artifact in sync with the latest training run.
What gets committed:
  - models/accessguard_model.pkl
  - mlops/monitoring/metrics.json
Commit message: "🤖 [MLOps] Auto-retrain: <timestamp>"
```

---

## 5. Running Locally

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the Streamlit app
```bash
cd AccessGuard_MLOps
streamlit run accessguard/main.py
```

### Manually retrain the model
```bash
python scripts/retrain.py
# With drift check:
python scripts/retrain.py --check-drift
```

### Run tests
```bash
pytest tests/ -v
```

### Simulate a registration trigger (no UI needed)
```python
from mlops.pipeline.trigger_pipeline import on_new_registration
on_new_registration("test_user")
```

### Check drift against last training metrics
```bash
python mlops/monitoring/monitor_drift.py
```

---

## 6. GitHub Secrets to Configure

Go to your repository → **Settings → Secrets and variables → Actions → New repository secret**

| Secret name | Value | Required? |
|-------------|-------|-----------|
| `SLACK_WEBHOOK_URL` | Your Slack incoming webhook URL | Optional — enables Slack alerts on drift or successful deploy |

The `GITHUB_TOKEN` secret is automatically provided by GitHub Actions — no setup needed.

### Enable auto-push from in-app trigger (optional)

If you want the Streamlit app itself to push the model after retraining (not just GitHub Actions), set this environment variable wherever the app runs:

```bash
export ENABLE_GIT_PUSH=true
```

The runner must have write access to the repository (e.g. via a deploy key or PAT stored in env).

---

## 7. Monitoring & Drift Detection

### metrics.json format

Every training run appends one entry:
```json
[
  {
    "timestamp": "2025-10-07T18:55:00",
    "accuracy": 0.8750,
    "f1_weighted": 0.8612,
    "roc_auc": 0.9100,
    "confusion_matrix": [[45, 5], [8, 42]],
    "classes": ["0.0", "1.0"],
    "n_estimators": 100
  }
]
```

### Viewing metrics history

```python
import json
with open("mlops/monitoring/metrics.json") as f:
    history = json.load(f)
for run in history:
    print(run["timestamp"], "acc:", run["accuracy"], "f1:", run["f1_weighted"])
```

### Alerts log

`data/alerts.log` contains timestamped HIGH/MEDIUM/LOW security events. Slack notifications go to `SLACK_WEBHOOK_URL` if configured.

---

## 8. Extending the Pipeline

### Retrain on every N registrations (not just 1)
```bash
export RETRAIN_EVERY_N_USERS=5
```

### Add more features to the model

Edit `mlops/pipeline/train_pipeline.py`:
1. Add the feature column in `preprocess()`
2. Add it to the `features` list in `train()`
3. Rerun or push — the CI pipeline will pick it up automatically

### Add a new drift threshold

Edit `THRESHOLDS` dict in `mlops/monitoring/monitor_drift.py`:
```python
THRESHOLDS = {
    "accuracy":    0.75,   # raise bar
    "f1_weighted": 0.70,
    "roc_auc":     0.80,   # add AUC check
}
```

### Schedule retraining more frequently

Edit the `cron` in `.github/workflows/mlops_pipeline.yml`:
```yaml
schedule:
  - cron: "0 */6 * * *"   # every 6 hours instead of nightly
```
