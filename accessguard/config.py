import os

# Data directory
DATA_DIR = "data"
VIDEO_DIR = os.path.join(DATA_DIR, "videos")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# File paths
USERS_CSV = os.path.join(DATA_DIR, "users.csv")
LOGINS_CSV = os.path.join(DATA_DIR, "logins.csv")
MODEL_FILE = os.path.join(DATA_DIR, "accessguard_model.pkl")
