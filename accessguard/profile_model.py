# profile_model.py
import os
import pandas as pd
from datetime import datetime
from config import (
    LOGINS_CSV,
)
from utils import get_geolocation


def compute_user_profile(username, min_rows=5):
    """
    Compute simple profile for a user:
      - mean_hour, std_hour (based on Timestamp)
      - last_geolocation (from most recent IP)
    Save profile as a JSON/pickle (optional) or return dict.
    """
    if not os.path.exists(LOGINS_CSV):
        return None

    logins = pd.read_csv(LOGINS_CSV)
    if logins.empty:
        return None

    user_rows = logins[logins["Username"] == username]
    if len(user_rows) < min_rows:
        return None

    # parse timestamps, get hours
    user_rows["Timestamp"] = pd.to_datetime(user_rows["Timestamp"], errors="coerce")
    user_rows = user_rows.dropna(subset=["Timestamp"])
    if user_rows.empty:
        return None

    hours = user_rows["Timestamp"].dt.hour
    mean_hour = hours.mean()
    std_hour = hours.std(ddof=0)

    # most recent IP -> geolocation
    last_row = user_rows.sort_values("Timestamp", ascending=False).iloc[0]
    last_ip = last_row.get("IP", None)
    last_geo = get_geolocation(last_ip) if last_ip else "Unknown"

    profile = {
        "username": username,
        "mean_hour": float(mean_hour),
        "std_hour": float(std_hour) if not pd.isna(std_hour) else 0.0,
        "last_geolocation": last_geo,
        "last_ip": last_ip,
        "computed_at": datetime.now().isoformat(),
        "num_rows": len(user_rows),
    }

    return profile


def get_user_profile(username):
    """Convenience wrapper: returns computed profile (no caching)."""
    try:
        return compute_user_profile(username)
    except Exception:
        return None
