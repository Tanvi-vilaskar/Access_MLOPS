import json
from datetime import datetime

import pandas as pd
import streamlit as st

from accessguard.auth import check_password, register_user
from accessguard.chatbot import chatbot_response
from accessguard.data_handler import (
    ensure_csvs_exist,
    load_logins,
    load_users,
    save_logins,
)
from accessguard.mfa_verification import check_face_in_image
from accessguard.model import train_login_model
from accessguard.risk import predict_login
from accessguard.utils import (
    get_browser_info,
    get_device_info,
    get_hour,
    get_ip,
)


# ------------------ Helper Function ------------------
def log_login_attempt(
    username,
    ip,
    device,
    browser,
    mfa_enabled,
    outcome,
    risk_score=None,
    risk_decision=None,
    reasons=None,
    timestamp=None,
):
    """Save login attempt in logins.csv including risk score, decision, and reasons"""
    if reasons is None:
        reasons = []
    elif not isinstance(reasons, list):
        try:
            reasons = json.loads(reasons)
        except json.JSONDecodeError:
            reasons = []

    new_row = {
        "Username": username,
        "IP": ip,
        "Device": device,
        "Browser": browser,
        "Timestamp": timestamp if timestamp else datetime.now().isoformat(),
        "MFA Enabled": mfa_enabled,
        "Outcome": outcome,
        "Risk Score": risk_score,
        "Risk Decision": risk_decision,
        "Reasons": json.dumps(reasons),
    }

    logins = load_logins()
    logins = pd.concat([logins, pd.DataFrame([new_row])], ignore_index=True)
    save_logins(logins)


# ------------------ Main App ------------------
def main():
    st.title("🔐 AccessGuard Demo")

    ensure_csvs_exist()

    page = st.sidebar.selectbox(
        "Choose Page",
        ["Register", "Login", "Train Model", "Admin", "🤖 Chatbot"],
    )

    # ---------------- Session State ----------------
    if "show_mfa_camera" not in st.session_state:
        st.session_state.show_mfa_camera = False

    if "login_info" not in st.session_state:
        st.session_state.login_info = None

    if "mfa_verified" not in st.session_state:
        st.session_state.mfa_verified = False

    if "registration_done" not in st.session_state:
        st.session_state.registration_done = False

    # ---------------- REGISTER ----------------
    if page == "Register":
        st.subheader("📝 Register New User")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        mfa_enabled = st.checkbox("Enable MFA?")

        st.info("📸 Please capture a photo of your face to complete registration.")

        reg_photo = st.camera_input(
            "Take a photo for face registration",
            key="reg_camera",
        )

        if st.button("Register") and not st.session_state.registration_done:
            if username.strip() == "" or password.strip() == "":
                st.warning("Enter both username and password")

            elif reg_photo is None:
                st.warning("📸 Please capture your face photo before registering.")

            else:
                with st.spinner("Detecting face..."):
                    face_found = check_face_in_image(reg_photo)

                if not face_found:
                    st.error(
                        "❌ No face detected in the photo. Please retake and try again."
                    )

                else:
                    uid = register_user(
                        username,
                        password,
                        mfa_enabled,
                    )

                    if uid is None:
                        st.error(
                            "❌ Registration failed. User already exists or error occurred."
                        )

                    else:
                        st.success(f"✅ User {username} registered with ID {uid}")
                        st.session_state.registration_done = True

    # ---------------- LOGIN ----------------
    elif page == "Login":
        st.subheader("🔑 Login")

        username = st.text_input("Enter username")
        password = st.text_input("Enter password", type="password")

        st.info("📸 Please capture a photo of your face to verify your identity.")

        login_photo = st.camera_input(
            "Take a photo to verify your face",
            key="login_camera",
        )

        if st.button("Login", key="initial_login"):
            st.session_state.show_mfa_camera = False
            st.session_state.login_info = None
            st.session_state.mfa_verified = False

            # ---------------- Face Verification ----------------
            if login_photo is None:
                st.warning("📸 Please capture your face photo before logging in.")
                st.stop()

            with st.spinner("Verifying face..."):
                face_ok = check_face_in_image(login_photo)

            if not face_ok:
                st.error(
                    "❌ Face not detected. Please retake your photo and try again."
                )

                log_login_attempt(
                    username,
                    get_ip(),
                    get_device_info(),
                    get_browser_info(),
                    0,
                    outcome=1,
                    risk_score=1.0,
                    risk_decision="BLOCK",
                    reasons=["Face not detected at login"],
                )

                st.stop()

            users = load_users()

            if username not in users["Username"].values:
                st.error("❌ User not registered!")

                log_login_attempt(
                    username,
                    get_ip(),
                    get_device_info(),
                    get_browser_info(),
                    0,
                    outcome=1,
                    risk_score=1.0,
                    risk_decision="BLOCK",
                    reasons=["User not registered"],
                )

            else:
                user_row = users.loc[users["Username"] == username].iloc[0]

                if not check_password(
                    password,
                    user_row["Password"],
                ):
                    st.error("❌ Incorrect password!")

                    log_login_attempt(
                        username,
                        get_ip(),
                        get_device_info(),
                        get_browser_info(),
                        int(user_row["MFA Enabled"]),
                        outcome=1,
                        risk_score=1.0,
                        risk_decision="BLOCK",
                        reasons=["Incorrect password"],
                    )

                else:
                    ip = get_ip()
                    device = get_device_info()
                    browser = get_browser_info()
                    hour = get_hour()

                    mfa_enabled = int(user_row["MFA Enabled"])

                    risk_score, decision, reasons = predict_login(
                        username,
                        ip,
                        device,
                        browser,
                        hour,
                        mfa_enabled,
                        registered_ip=user_row["IP"],
                        registered_device=user_row["Device"],
                        registered_browser=user_row["Browser"],
                    )

                    st.info(f"**Risk Decision:** {decision}")
                    st.info(f"**Risk Score:** {risk_score:.2f}")

                    if reasons:
                        st.info("⚠️ Risk assessment details:")

                        for reason in reasons:
                            st.write("-", reason)

                    # ---------------- MFA REQUIRED ----------------
                    if "ALLOW with MFA" in decision:
                        timestamp = datetime.now().isoformat()

                        st.session_state.login_info = {
                            "username": username,
                            "ip": ip,
                            "device": device,
                            "browser": browser,
                            "mfa_enabled": mfa_enabled,
                            "risk_score": risk_score,
                            "risk_decision": decision,
                            "reasons": reasons,
                            "timestamp": timestamp,
                        }

                        log_login_attempt(
                            username,
                            ip,
                            device,
                            browser,
                            mfa_enabled,
                            outcome=None,
                            risk_score=risk_score,
                            risk_decision=decision,
                            reasons=reasons,
                            timestamp=timestamp,
                        )

                        st.warning(
                            "⚠️ Medium risk detected — MFA verification required."
                        )

                        st.session_state.show_mfa_camera = True

                    # ---------------- DIRECT ALLOW ----------------
                    elif "ALLOW" in decision:
                        log_login_attempt(
                            username,
                            ip,
                            device,
                            browser,
                            mfa_enabled,
                            outcome=0,
                            risk_score=risk_score,
                            risk_decision=decision,
                            reasons=reasons,
                        )

                        st.success(f"✅ Login allowed. Welcome, {username}!")

                    # ---------------- BLOCK ----------------
                    elif "BLOCK" in decision:
                        log_login_attempt(
                            username,
                            ip,
                            device,
                            browser,
                            mfa_enabled,
                            outcome=1,
                            risk_score=risk_score,
                            risk_decision=decision,
                            reasons=reasons,
                        )

                        st.error("❌ High risk detected. Access denied.")

        # ---------------- MFA FLOW ----------------
        if st.session_state.show_mfa_camera and not st.session_state.mfa_verified:
            st.subheader("Step 2: Face Verification")

            st.info("Please capture a photo of your face using the camera below.")

            mfa_photo = st.camera_input(
                "Capture your face photo",
                key="mfa_camera",
            )

            if mfa_photo is not None:
                info = st.session_state.login_info

                with st.spinner("Analyzing image for face detection..."):
                    success = check_face_in_image(mfa_photo)

                updated_decision = "ALLOW" if success else "BLOCK"
                outcome = 0 if success else 1

                logins = load_logins()

                idx = logins[
                    (logins["Username"] == info["username"])
                    & (logins["Risk Decision"] == "ALLOW with MFA")
                    & (logins["Timestamp"] == info["timestamp"])
                ].index

                if not idx.empty:
                    logins.loc[idx[0], "Risk Decision"] = updated_decision

                    logins.loc[idx[0], "Outcome"] = outcome

                    save_logins(logins)

                else:
                    log_login_attempt(
                        info["username"],
                        info["ip"],
                        info["device"],
                        info["browser"],
                        info["mfa_enabled"],
                        outcome=outcome,
                        risk_score=info["risk_score"],
                        risk_decision=updated_decision,
                        reasons=info["reasons"],
                        timestamp=info["timestamp"],
                    )

                if success:
                    st.success(
                        f"✅ MFA verification completed — login allowed. Welcome, {info['username']}!"
                    )

                else:
                    st.error("❌ MFA verification failed. Access denied.")

                st.session_state.mfa_verified = True
                st.session_state.show_mfa_camera = False

    # ---------------- TRAIN MODEL ----------------
    elif page == "Train Model":
        st.subheader("🛠️ Train ML Model")

        if st.button("Train"):
            model, encoders = train_login_model()

            if model:
                st.success("✅ Model trained & saved successfully.")

            else:
                st.warning("⚠️ Not enough data to train model (need ≥10 rows).")

    # ---------------- ADMIN ----------------
    elif page == "Admin":
        st.subheader("🔎 Admin Dashboard")

        logins = load_logins()

        if logins.empty:
            st.info("No login attempts found.")

        else:

            def normalize_decision(x):
                if pd.isna(x):
                    return "UNKNOWN"

                x = str(x).lower().strip()

                if "allow with mfa" in x:
                    return "ALLOW with MFA"

                if "allow" in x:
                    return "ALLOW"

                if "block" in x:
                    return "BLOCK"

                return "UNKNOWN"

            logins["Risk Decision"] = logins["Risk Decision"].apply(normalize_decision)

            st.write("📊 Login Decisions Summary:")

            decision_counts = logins["Risk Decision"].value_counts()

            st.write(decision_counts)

            st.write("**Risk Decision Distribution:**")
            st.bar_chart(decision_counts)

            st.write("**Most Frequent Blocked Users:**")

            blocked_users = logins[logins["Outcome"] == 1]["Username"].value_counts()

            if not blocked_users.empty:
                st.bar_chart(blocked_users)

            else:
                st.info("No blocked users found.")

            st.write("**Blocked Attempts Over Time:**")

            blocked_time = logins[logins["Outcome"] == 1].copy()

            if not blocked_time.empty:
                blocked_time["Date"] = pd.to_datetime(blocked_time["Timestamp"]).dt.date

                blocked_per_day = blocked_time.groupby("Date").size()

                st.line_chart(blocked_per_day)

            else:
                st.info("No blocked attempts over time.")

            st.write("**Recent Blocked Users (including MFA failures):**")

            blocked_recent = logins[logins["Outcome"] == 1].sort_values(
                "Timestamp",
                ascending=False,
            )

            if not blocked_recent.empty:
                st.dataframe(
                    blocked_recent[
                        [
                            "Username",
                            "Timestamp",
                            "Risk Decision",
                            "Reasons",
                        ]
                    ]
                    .head(10)
                    .rename(columns={"Reasons": "Details"})
                )

            else:
                st.info("No recent blocked users found.")

    # ---------------- CHATBOT ----------------
    elif page == "🤖 Chatbot":
        st.subheader("🤖 Login History Chatbot")

        st.markdown(
            "Ask me anything about login history, blocked attempts, risk scores, and more!"
        )

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": (
                        "👋 Hi! I'm the AccessGuard Login Bot.\n\n"
                        "I can tell you about login history, blocked attempts, "
                        "risk scores, devices, IPs and more.\n\n"
                        "Type help to see all questions I can answer!"
                    ),
                }
            ]

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about login history..."):
            st.session_state.chat_messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chatbot_response(prompt)

                st.markdown(response)

            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

        st.divider()

        st.markdown("**💡 Quick Questions:**")

        cols = st.columns(3)

        quick_questions = [
            "Show login summary",
            "Last 5 logins",
            "Blocked attempts",
            "Show all users",
            "Risk score stats",
            "Logins today",
        ]

        for i, question in enumerate(quick_questions):
            col = cols[i % 3]

            if col.button(question, key=f"quick_{i}"):
                st.session_state.chat_messages.append(
                    {
                        "role": "user",
                        "content": question,
                    }
                )

                response = chatbot_response(question)

                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                    }
                )

                st.rerun()


if __name__ == "__main__":
    main()
