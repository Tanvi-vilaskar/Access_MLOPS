# utils.py

import socket
import platform
from datetime import datetime, timedelta

import pandas as pd
import requests


# =============================
# 📍 1. System Info Functions
# =============================
def get_ip():
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "0.0.0.0"


def get_device_info():
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def get_browser_info():
    return "Streamlit-App"


def get_hour():
    return datetime.now().hour


# =============================
# 🌍 2. Get Geolocation by IP
# =============================
def get_geolocation(ip_address: str) -> dict:
    """
    Fetch geolocation info using ipapi.
    """
    try:
        response = requests.get(f"https://ipapi.co/{ip_address}/json/", timeout=5)

        if response.status_code == 200:
            data = response.json()
            return {
                "ip": ip_address,
                "city": data.get("city", "Unknown"),
                "region": data.get("region", "Unknown"),
                "country": data.get("country_name", "Unknown"),
                "latitude": data.get("latitude"),
                "longitude": data.get("longitude"),
            }

        return {"ip": ip_address, "error": "API request failed"}

    except requests.RequestException as error:
        return {"ip": ip_address, "error": str(error)}


# ==================================
# 🕒 3. Login Attempts in Last Hour
# ==================================
def login_attempts_in_last_hour(
    username: str, log_path: str = "data/logins.csv"
) -> int:
    """
    Count login attempts by user in last 1 hour.
    """
    try:
        df = pd.read_csv(log_path)

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        one_hour_ago = datetime.now() - timedelta(hours=1)

        recent_attempts = df[
            (df["username"] == username) & (df["timestamp"] > one_hour_ago)
        ]

        return len(recent_attempts)

    except FileNotFoundError:
        return 0
    except Exception:
        return 0


# =========================
# 📁 4. Load Logins Dataset
# =========================
def load_logins(
    log_path: str = "data/logins.csv",
) -> pd.DataFrame:
    """
    Load login logs safely.
    """
    try:
        return pd.read_csv(log_path)

    except FileNotFoundError:
        columns = [
            "timestamp",
            "username",
            "ip",
            "device",
            "browser",
            "risk_score",
            "status",
        ]
        return pd.DataFrame(columns=columns)
