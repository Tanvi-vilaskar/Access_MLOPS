# user_models.py
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from data_handler import load_logins
from config import DATA_DIR

USER_MODEL_DIR = os.path.join(DATA_DIR, "user_models")
os.makedirs(USER_MODEL_DIR, exist_ok=True)


def train_user_model(username, min_rows=50):
    logins = load_logins()
    user_rows = logins[logins["Username"] == username]
    if len(user_rows) < min_rows:
        return None

    # Create features: IP, Device, Browser, Hour
    user_rows["Timestamp"] = pd.to_datetime(user_rows["Timestamp"], errors="coerce")
    user_rows = user_rows.dropna(subset=["Timestamp"])
    user_rows["Hour"] = user_rows["Timestamp"].dt.hour

    X = user_rows[["IP", "Device", "Browser", "Hour"]].astype(str)
    encoders = {}
    for col in ["IP", "Device", "Browser"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    y = user_rows["Outcome"].astype(int)  # existing label: 0 = success, 1 = blocked
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    model_path = os.path.join(USER_MODEL_DIR, f"{username}.joblib")
    joblib.dump((model, encoders), model_path)
    return model_path


def load_user_model(username):
    model_path = os.path.join(USER_MODEL_DIR, f"{username}.joblib")
    if not os.path.exists(model_path):
        return None, None
    return joblib.load(model_path)
