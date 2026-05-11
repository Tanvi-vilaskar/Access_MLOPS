import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

from data_handler import load_logins
from config import MODEL_FILE


# ============================================================
# 1. PREPROCESSING
# ============================================================
def _preprocess_logins(logins: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans dataset and creates features:
    - LoginHour
    - User_New_IP
    """

    # Remove missing target
    logins = logins.dropna(subset=["Outcome"])

    # Remove rows with missing key features
    logins = logins.dropna(
        subset=["IP", "Device", "Browser", "MFA Enabled", "Username"]
    )

    # Convert MFA Enabled to numeric
    mfa_numeric = pd.to_numeric(logins["MFA Enabled"], errors="coerce")
    logins = logins[mfa_numeric.notna()].copy()
    logins["MFA Enabled"] = mfa_numeric.dropna().astype(int)

    # Extract login hour
    logins["Timestamp"] = pd.to_datetime(logins["Timestamp"])
    logins["LoginHour"] = logins["Timestamp"].dt.hour

    # Detect new IP per user
    known_ips = {}
    logins["User_New_IP"] = 0

    logins_sorted = logins.sort_values(by="Timestamp").reset_index(drop=True)

    for index, row in logins_sorted.iterrows():
        user = row["Username"]
        ip = row["IP"]

        if user not in known_ips:
            known_ips[user] = set()

        if ip not in known_ips[user]:
            logins_sorted.loc[index, "User_New_IP"] = 1

        known_ips[user].add(ip)

    return logins_sorted


# ============================================================
# 2. TRAIN MODEL
# ============================================================
def train_login_model():
    """
    Train RandomForest model and save it.
    """

    logins = load_logins()
    logins = _preprocess_logins(logins)

    if len(logins) < 10:
        print("Not enough data to train model.")
        return None, None

    # Features
    features = [
        "IP",
        "Device",
        "Browser",
        "MFA Enabled",
        "LoginHour",
        "User_New_IP",
    ]

    X = logins[features].astype(str)

    # Encode categorical columns
    encoders = {}
    for col in ["IP", "Device", "Browser"]:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])
        encoders[col] = encoder

    # Convert numeric features
    X["MFA Enabled"] = logins["MFA Enabled"].astype(int)
    X["LoginHour"] = logins["LoginHour"].astype(int)
    X["User_New_IP"] = logins["User_New_IP"].astype(int)

    # Target
    y_str = logins["Outcome"].astype(str).fillna("Unknown")
    outcome_encoder = LabelEncoder()
    y = outcome_encoder.fit_transform(y_str)
    encoders["Outcome"] = outcome_encoder

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    # Evaluation
    display_confusion_matrix(model, X_test, y_test, outcome_encoder)
    plot_confusion_matrix(model, X_test, y_test, outcome_encoder)

    # Save model
    joblib.dump((model, encoders), MODEL_FILE)
    print(f"\nModel saved at: {MODEL_FILE}")

    return model, encoders


# ============================================================
# 3. TEXT CONFUSION MATRIX
# ============================================================
def display_confusion_matrix(model, X_test, y_test, outcome_encoder):
    print("\n--- Model Evaluation ---")

    y_pred = model.predict(X_test)
    labels = outcome_encoder.classes_
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")

    # FIXED E741 here (label instead of l)
    header = [""] + [f"Pred {label}" for label in labels]
    print(" | ".join(f"{h:<8}" for h in header))
    print("-" * (9 * len(header)))

    for i, true_label in enumerate(labels):
        row = [f"True {true_label}"] + [f"{cm[i, j]:<8}" for j in range(len(labels))]
        print(" | ".join(row))

    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=labels, zero_division=0)
    print(report)


# ============================================================
# 4. PLOT CONFUSION MATRIX
# ============================================================
def plot_confusion_matrix(model, X_test, y_test, outcome_encoder):
    y_pred = model.predict(X_test)
    labels = outcome_encoder.classes_
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. LOAD MODEL
# ============================================================
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None, None
    return joblib.load(MODEL_FILE)


# ============================================================
# 6. RUN
# ============================================================
if __name__ == "__main__":
    train_login_model()
