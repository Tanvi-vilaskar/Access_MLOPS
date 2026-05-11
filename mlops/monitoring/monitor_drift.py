"""
monitor_drift.py
=================
Detects model performance drift by comparing the latest training
metrics against a configurable baseline threshold.

Run during CI or as a scheduled cron job:
    python mlops/monitoring/monitor_drift.py

Exit codes:
    0 — All metrics within acceptable range
    1 — Drift detected (CI will fail and alert team)
"""

import sys
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS_FILE = ROOT / "mlops" / "monitoring" / "metrics.json"

# ── Drift Thresholds (tune as needed) ──────────────────────────────────────────
THRESHOLDS = {
    "accuracy": 0.70,  # must be >= 70%
    "f1_weighted": 0.65,  # must be >= 65%
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MONITOR] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("drift_monitor")


def load_latest_metrics() -> dict:
    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_FILE}\n"
            "Run the training pipeline first."
        )
    with open(METRICS_FILE) as fh:
        data = json.load(fh)

    if isinstance(data, list):
        return data[-1]  # Most recent entry
    return data


def check_drift(metrics: dict) -> list[str]:
    """
    Returns a list of violation messages. Empty list = no drift.
    """
    violations = []
    for metric, threshold in THRESHOLDS.items():
        value = metrics.get(metric)
        if value is None:
            log.warning("Metric '%s' not found in metrics file — skipping.", metric)
            continue
        if value < threshold:
            msg = (
                f"DRIFT DETECTED — {metric}: {value:.4f} "
                f"is below threshold {threshold:.4f}"
            )
            violations.append(msg)
            log.error(msg)
        else:
            log.info("OK — %s: %.4f (>= %.4f)", metric, value, threshold)
    return violations


def generate_drift_report(metrics: dict, violations: list[str]) -> str:
    lines = [
        "=" * 55,
        "AccessGuard — Model Drift Report",
        f"Timestamp : {metrics.get('timestamp', 'N/A')}",
        "-" * 55,
        f"  Accuracy    : {metrics.get('accuracy', 'N/A')}",
        f"  F1 Weighted : {metrics.get('f1_weighted', 'N/A')}",
        f"  ROC AUC     : {metrics.get('roc_auc', 'N/A')}",
        "-" * 55,
    ]
    if violations:
        lines.append("STATUS: ❌ DRIFT DETECTED")
        for v in violations:
            lines.append(f"  • {v}")
    else:
        lines.append("STATUS: ✅ All metrics within acceptable range")
    lines.append("=" * 55)
    return "\n".join(lines)


def main() -> int:
    log.info("Running drift detection...")
    try:
        metrics = load_latest_metrics()
        violations = check_drift(metrics)
        report = generate_drift_report(metrics, violations)
        print("\n" + report + "\n")

        # Write report to file for CI artifact upload
        report_path = ROOT / "mlops" / "monitoring" / "drift_report.txt"
        report_path.write_text(report)

        return 1 if violations else 0

    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    except Exception as exc:
        log.exception("Unexpected error in drift monitor: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
