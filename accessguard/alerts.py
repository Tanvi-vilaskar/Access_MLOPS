# alerts.py
import datetime
import os
import requests

ALERT_LOG = "data/alerts.log"
SLACK_WEBHOOK = os.environ.get(
    "SLACK_WEBHOOK_URL"
)  # set this in your environment if you want Slack alerts


def _write_log(message: str):
    os.makedirs(os.path.dirname(ALERT_LOG), exist_ok=True)
    with open(ALERT_LOG, "a") as f:
        f.write(message + "\n")


def notify_manager(
    username: str, reason: str, video_path: str = None, severity: str = "HIGH"
):
    """
    Log locally and optionally send Slack notification.
    severity: "HIGH" / "MEDIUM" / "LOW"
    """
    ts = datetime.datetime.now().isoformat()
    message = f"[{ts}] ALERT [{severity}] {username} - {reason}"
    if video_path:
        message += f" (video: {video_path})"

    print(message)
    _write_log(message)

    # Slack notification (simple payload)
    if SLACK_WEBHOOK:
        try:
            payload = {
                "text": f"*AccessGuard Alert* - {severity}\n*User:* {username}\n*Reason:* {reason}\n*Time:* {ts}"
            }
            requests.post(SLACK_WEBHOOK, json=payload, timeout=5)
        except Exception as e:
            # Do not raise; just log failure
            _write_log(f"[{ts}] Failed to send Slack alert: {e}")
