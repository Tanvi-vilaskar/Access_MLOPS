"""
trigger_pipeline.py
====================
Called by accessguard/auth.py immediately after a new user is
successfully registered. It:

  1. Appends the registration event to a trigger log.
  2. Checks whether there is enough new data to warrant retraining.
  3. If yes, runs the full MLOps training pipeline in-process.
  4. Optionally commits & pushes the updated model artifact so that
     GitHub Actions picks it up (set ENABLE_GIT_PUSH=true in env).

Usage (called from auth.py):
    from mlops.pipeline.trigger_pipeline import on_new_registration
    on_new_registration(username)

Standalone test:
    python mlops/pipeline/trigger_pipeline.py
"""

import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRIGGER_LOG = ROOT / "mlops" / "monitoring" / "trigger.log"
METRICS_DIR = ROOT / "mlops" / "monitoring"
DATA_DIR = ROOT / "data"
LOGINS_CSV = DATA_DIR / "logins.csv"

# Minimum new registrations before retraining
RETRAIN_EVERY_N_USERS = int(os.getenv("RETRAIN_EVERY_N_USERS", "1"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TRIGGER] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(TRIGGER_LOG),
    ],
)
log = logging.getLogger("trigger")


def _count_registrations_since_last_train() -> int:
    """
    Count how many new registrations have occurred since the model
    was last trained (based on metrics.json timestamp).
    """
    import pandas as pd

    metrics_file = METRICS_DIR / "metrics.json"
    last_train_ts = None

    if metrics_file.exists():
        try:
            with open(metrics_file) as fh:
                history = json.load(fh)
            if isinstance(history, list) and history:
                last_train_ts = history[-1].get("timestamp")
            elif isinstance(history, dict):
                last_train_ts = history.get("timestamp")
        except Exception:
            pass

    users_csv = DATA_DIR / "users.csv"
    if not users_csv.exists():
        return 0

    users = pd.read_csv(users_csv)
    if "Registration Timestamp" not in users.columns:
        return len(users)  # assume all are new

    if last_train_ts is None:
        return len(users)

    users["Registration Timestamp"] = pd.to_datetime(
        users["Registration Timestamp"], errors="coerce"
    )
    last_train_dt = pd.to_datetime(last_train_ts)
    new_users = users[users["Registration Timestamp"] > last_train_dt]
    return len(new_users)


def _run_training_pipeline() -> bool:
    """Import and run the training pipeline. Returns True on success."""
    try:
        # Add project root to path so pipeline can import accessguard modules
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        from mlops.pipeline.train_pipeline import run_pipeline

        exit_code = run_pipeline()
        return exit_code == 0
    except Exception as exc:
        log.exception("Training pipeline raised an exception: %s", exc)
        return False


def _git_push_model():
    """
    Commit & push updated model artifact.
    Only runs when ENABLE_GIT_PUSH=true in environment.
    Requires the runner to have git credentials configured.
    """
    if os.getenv("ENABLE_GIT_PUSH", "false").lower() != "true":
        return

    model_path = ROOT / "models" / "accessguard_model.pkl"
    if not model_path.exists():
        log.warning("Model file not found; skipping git push.")
        return

    try:
        subprocess.run(
            ["git", "config", "user.email", "mlops-bot@accessguard.local"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "MLOps Bot"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "add", str(model_path), str(METRICS_DIR)],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                f"[auto] Retrain model after new registration {datetime.now().isoformat()}",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "push"], cwd=ROOT, check=True, capture_output=True)
        log.info("Model artifact pushed to repository.")
    except subprocess.CalledProcessError as exc:
        log.warning("Git push failed (non-fatal): %s", exc)


def on_new_registration(username: str):
    """
    Entry point called by auth.py after a successful user registration.

    Args:
        username: The username that was just registered.
    """
    ts = datetime.now().isoformat()
    log.info("New registration event: user='%s' at %s", username, ts)

    # Log the event
    event = {"event": "registration", "username": username, "timestamp": ts}
    with open(TRIGGER_LOG, "a") as fh:
        fh.write(json.dumps(event) + "\n")

    # Check whether retraining threshold is met
    new_count = _count_registrations_since_last_train()
    log.info(
        "New registrations since last train: %d (threshold: %d)",
        new_count,
        RETRAIN_EVERY_N_USERS,
    )

    if new_count >= RETRAIN_EVERY_N_USERS:
        log.info("Threshold met — starting retraining pipeline...")
        success = _run_training_pipeline()
        if success:
            log.info("Retraining completed successfully.")
            _git_push_model()
        else:
            log.error("Retraining FAILED. Check pipeline.log for details.")
    else:
        log.info(
            "Not enough new data yet (%d/%d). Skipping retrain.",
            new_count,
            RETRAIN_EVERY_N_USERS,
        )


if __name__ == "__main__":
    # Quick smoke test
    on_new_registration("test_user_cli")
