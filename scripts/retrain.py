"""
scripts/retrain.py
====================
Developer utility — run the full training pipeline manually.

Usage:
    python scripts/retrain.py
    python scripts/retrain.py --check-drift   # also run drift monitor after train
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Manually retrain the AccessGuard model."
    )
    parser.add_argument(
        "--check-drift",
        action="store_true",
        help="Run drift monitor after training and exit 1 if drift detected.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AccessGuard — Manual Retrain")
    print("=" * 60)

    from mlops.pipeline.train_pipeline import run_pipeline

    train_exit = run_pipeline()

    if train_exit != 0:
        print("\n❌ Training pipeline failed. Check logs.")
        sys.exit(1)

    print("\n✅ Training complete.")

    if args.check_drift:
        print("\nRunning drift monitor...")
        from mlops.monitoring.monitor_drift import main as drift_main

        drift_exit = drift_main()
        if drift_exit != 0:
            print("⚠️  Drift detected — see report above.")
            sys.exit(1)
        print("✅ No drift detected.")

    sys.exit(0)


if __name__ == "__main__":
    main()
