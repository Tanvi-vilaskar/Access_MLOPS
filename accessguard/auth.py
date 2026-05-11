# accessguard/auth.py  — updated with MLOps pipeline trigger
import pandas as pd
import bcrypt
from datetime import datetime
from data_handler import load_users, save_users
from utils import get_ip, get_device_info, get_browser_info

try:
    from mlops.pipeline.trigger_pipeline import on_new_registration

    _MLOPS_AVAILABLE = True
except ImportError:
    _MLOPS_AVAILABLE = False


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def check_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def register_user(username: str, password: str, mfa_enabled: bool):
    users = load_users()
    if username in users["Username"].values:
        print(f"[Auth] Username '{username}' already exists.")
        return None

    video_path = None
    try:
        from video import capture_video

        video_path = capture_video(username)
        if video_path is None:
            print("[Auth] Face not detected. Registration failed.")
            return None
    except Exception as e:
        print(f"[Auth] Video capture skipped (headless): {e}")

    if users.empty:
        new_uid = 1000
    else:
        uid_col = "UID" if "UID" in users.columns else "User ID"
        users["UID_numeric"] = pd.to_numeric(
            users.get(uid_col, pd.Series(dtype=float)), errors="coerce"
        )
        new_uid = int(users["UID_numeric"].max()) + 1

    timestamp = datetime.now().isoformat()
    new_row = {
        "User ID": new_uid,
        "Username": username,
        "Password": hash_password(password),
        "Video File": video_path or "",
        "Registered At": timestamp,
        "MFA Enabled": int(mfa_enabled),
        "IP": get_ip(),
        "Device": get_device_info(),
        "Browser": get_browser_info(),
        "UID": new_uid,
        "Video Path": video_path or "",
        "Registration Timestamp": timestamp,
        "UID_numeric": new_uid,
    }
    users = pd.concat([users, pd.DataFrame([new_row])], ignore_index=True)
    save_users(users)
    print(f"[Auth] User '{username}' registered (UID={new_uid}).")

    # ── Trigger MLOps retrain pipeline ────────────────────────────────────────
    if _MLOPS_AVAILABLE:
        try:
            on_new_registration(username)
        except Exception as exc:
            print(f"[Auth] MLOps trigger error (non-fatal): {exc}")
    return new_uid
