# chatbot.py — Login History Chatbot Engine (v2)
import re
import json
import pandas as pd
from datetime import datetime
from data_handler import load_logins, load_users

# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────


def _load_logins() -> pd.DataFrame:
    df = load_logins()
    if df.empty:
        return df
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if "Outcome" in df.columns:
        df["Outcome"] = pd.to_numeric(df["Outcome"], errors="coerce")
    if "Risk Score" in df.columns:
        df["Risk Score"] = pd.to_numeric(df["Risk Score"], errors="coerce")
    return df


def _load_users() -> pd.DataFrame:
    try:
        return load_users()
    except Exception:
        return pd.DataFrame()


def _fmt_ts(ts) -> str:
    try:
        return pd.Timestamp(ts).strftime("%d %b %Y  %H:%M")
    except Exception:
        return str(ts)


def _outcome_label(val) -> str:
    if pd.isna(val):
        return "⏳ Pending"
    return "✅ Allowed" if int(val) == 0 else "❌ Blocked"


def _known_users(df: pd.DataFrame):
    return sorted(df["Username"].dropna().unique().tolist()) if not df.empty else []


def _extract_username(text: str, df: pd.DataFrame) -> str | None:
    for u in _known_users(df):
        if u.lower() in text.lower():
            return u
    return None


def _extract_n(text: str, default: int = 5) -> int:
    m = re.search(r"\b(\d+)\b", text)
    return int(m.group(1)) if m else default


def _parse_reasons(raw) -> str:
    """Safely parse reason string to readable text."""
    if pd.isna(raw) or str(raw).strip() in ("", "[]", "nan"):
        return "—"
    try:
        items = json.loads(str(raw))
        if isinstance(items, list) and items:
            return "; ".join(str(i) for i in items)
    except Exception:
        pass
    cleaned = str(raw).strip("[]\"'")
    return cleaned if cleaned else "—"


# ─────────────────────────────────────────────
# Intent handlers
# ─────────────────────────────────────────────


def _intent_summary(df: pd.DataFrame, username: str | None) -> str:
    if df.empty:
        return "📭 No login records found yet."
    sub = df[df["Username"] == username] if username else df
    if sub.empty:
        return f"❌ No records found for **{username}**."

    total = len(sub)
    allowed = int((sub["Outcome"] == 0).sum()) if "Outcome" in sub.columns else "N/A"
    blocked = int((sub["Outcome"] == 1).sum()) if "Outcome" in sub.columns else "N/A"
    label = f" for **{username}**" if username else " (all users)"

    avg_risk = ""
    if "Risk Score" in sub.columns:
        mean = sub["Risk Score"].mean()
        if not pd.isna(mean):
            avg_risk = f"\n- 🎯 Average risk score : **{mean:.2f}**"

    first_ts = sub["Timestamp"].min() if "Timestamp" in sub.columns else None
    last_ts = sub["Timestamp"].max() if "Timestamp" in sub.columns else None
    period = ""
    if (
        first_ts is not None
        and last_ts is not None
        and pd.notna(first_ts)
        and pd.notna(last_ts)
    ):
        period = f"\n- 📅 Period  : **{_fmt_ts(first_ts)}** → **{_fmt_ts(last_ts)}**"

    return (
        f"📊 **Login summary{label}:**\n"
        f"- 🔢 Total attempts : **{total}**\n"
        f"- ✅ Allowed        : **{allowed}**\n"
        f"- ❌ Blocked        : **{blocked}**" + avg_risk + period
    )


def _intent_recent(df: pd.DataFrame, username: str | None, n: int) -> str:
    if df.empty:
        return "📭 No login records found yet."
    sub = df[df["Username"] == username] if username else df
    if sub.empty:
        return f"❌ No records found for **{username}**."

    sub = sub.sort_values("Timestamp", ascending=False).head(n)
    label = f" for **{username}**" if username else " (all users)"
    lines = [f"🕐 **Last {n} login attempts{label}:**\n"]
    lines.append("| # | Time | User | Status | IP | Risk |")
    lines.append("|---|------|------|--------|----|------|")
    for i, (_, row) in enumerate(sub.iterrows(), 1):
        ts = _fmt_ts(row.get("Timestamp", ""))
        user = row.get("Username", "—")
        outcome = _outcome_label(row.get("Outcome"))
        ip = row.get("IP", "—")
        risk = row.get("Risk Score", "")
        risk_str = f"{risk:.0f}" if pd.notna(risk) and risk != "" else "—"
        lines.append(f"| {i} | {ts} | **{user}** | {outcome} | `{ip}` | {risk_str} |")
    return "\n".join(lines)


def _intent_blocked(df: pd.DataFrame, username: str | None) -> str:
    if df.empty:
        return "📭 No login records found yet."

    sub = df if not username else df[df["Username"] == username]
    if "Outcome" not in sub.columns:
        return "⚠️ Outcome data not available."

    blocked = sub[sub["Outcome"] == 1].sort_values("Timestamp", ascending=False)
    if blocked.empty:
        who = f" for **{username}**" if username else ""
        return f"✅ No blocked login attempts{who}!"

    label = f" for **{username}**" if username else " (all users)"
    total = len(blocked)
    lines = [f"🚫 **Blocked attempts{label} — {total} total:**\n"]
    lines.append("| # | Time | User | IP | Risk | Reason |")
    lines.append("|---|------|------|----|------|--------|")

    for i, (_, row) in enumerate(blocked.head(15).iterrows(), 1):
        ts = _fmt_ts(row.get("Timestamp", ""))
        user = row.get("Username", "—")
        ip = row.get("IP", "—")
        risk = row.get("Risk Score", "")
        risk_s = f"{risk:.0f}" if pd.notna(risk) and risk != "" else "—"
        reason = _parse_reasons(row.get("Reasons", ""))
        # Truncate long reasons
        if len(reason) > 50:
            reason = reason[:47] + "..."
        lines.append(f"| {i} | {ts} | **{user}** | `{ip}` | {risk_s} | {reason} |")

    if total > 15:
        lines.append(f"\n_...and {total - 15} more blocked attempts._")
    return "\n".join(lines)


def _intent_last_login(df: pd.DataFrame, username: str | None) -> str:
    if df.empty:
        return "📭 No login records found yet."
    sub = df[df["Username"] == username] if username else df
    if sub.empty:
        return f"❌ No records found for **{username}**."

    row = sub.sort_values("Timestamp", ascending=False).iloc[0]
    ts = _fmt_ts(row.get("Timestamp", ""))
    ip = row.get("IP", "—")
    device = row.get("Device", "—")
    browser = row.get("Browser", "—")
    outcome = _outcome_label(row.get("Outcome"))
    risk = row.get("Risk Score", "")
    risk_str = (
        f"\n- 🎯 Risk score : **{risk:.2f}**" if pd.notna(risk) and risk != "" else ""
    )
    decision = row.get("Risk Decision", "")
    dec_str = (
        f"\n- 🔐 Decision   : **{decision}**"
        if pd.notna(decision) and str(decision).strip()
        else ""
    )
    who = f"**{username}**" if username else f"**{row.get('Username','unknown')}**"

    return (
        f"🔍 **Last login for {who}:**\n"
        f"- 🕐 Time    : **{ts}**\n"
        f"- 🌐 IP      : `{ip}`\n"
        f"- 💻 Device  : **{device}**\n"
        f"- 🌍 Browser : **{browser}**\n"
        f"- Status    : {outcome}" + risk_str + dec_str
    )


def _intent_devices(df: pd.DataFrame, username: str | None) -> str:
    if df.empty:
        return "📭 No login records found yet."
    sub = df[df["Username"] == username] if username else df
    if sub.empty:
        return f"❌ No records found for **{username}**."

    label = f" for **{username}**" if username else " (all users)"
    devices = sub["Device"].value_counts() if "Device" in sub.columns else pd.Series()
    if devices.empty:
        return f"No device data found{label}."

    lines = [f"💻 **Devices used{label}:**\n"]
    lines.append("| Device | Logins |")
    lines.append("|--------|--------|")
    for dev, cnt in devices.items():
        lines.append(f"| {dev} | {cnt} |")
    return "\n".join(lines)


def _intent_ip(df: pd.DataFrame, username: str | None) -> str:
    if df.empty:
        return "📭 No login records found yet."
    sub = df[df["Username"] == username] if username else df
    if sub.empty:
        return f"❌ No records found for **{username}**."

    label = f" for **{username}**" if username else " (all users)"
    ips = sub["IP"].value_counts() if "IP" in sub.columns else pd.Series()
    if ips.empty:
        return f"No IP data found{label}."

    lines = [f"🌐 **IP addresses used{label}:**\n"]
    lines.append("| IP Address | Logins |")
    lines.append("|------------|--------|")
    for ip, cnt in ips.head(10).items():
        lines.append(f"| `{ip}` | {cnt} |")
    return "\n".join(lines)


def _intent_risk(df: pd.DataFrame, username: str | None) -> str:
    if df.empty:
        return "📭 No login records found yet."
    if "Risk Score" not in df.columns:
        return "⚠️ Risk score data is not available."
    sub = df[df["Username"] == username] if username else df
    if sub.empty:
        return f"❌ No records found for **{username}**."

    label = f" for **{username}**" if username else " (all users)"
    scores = sub["Risk Score"].dropna()
    if scores.empty:
        return f"No risk score data available{label}."

    high_risk = (
        sub[sub["Risk Score"] >= 60] if "Risk Score" in sub.columns else pd.DataFrame()
    )
    med_risk = (
        sub[(sub["Risk Score"] >= 30) & (sub["Risk Score"] < 60)]
        if "Risk Score" in sub.columns
        else pd.DataFrame()
    )

    lines = [f"🎯 **Risk score analysis{label}:**\n"]
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Average score | **{scores.mean():.2f}** |")
    lines.append(f"| Highest score | **{scores.max():.2f}** |")
    lines.append(f"| Lowest score  | **{scores.min():.2f}** |")
    lines.append(f"| 🔴 High risk (≥60) | **{len(high_risk)}** attempts |")
    lines.append(f"| 🟡 Medium risk (30-59) | **{len(med_risk)}** attempts |")
    lines.append(
        f"| 🟢 Low risk (<30) | **{len(sub) - len(high_risk) - len(med_risk)}** attempts |"
    )
    return "\n".join(lines)


def _intent_users(df: pd.DataFrame) -> str:
    """Show ALL registered users with login counts from both users.csv and logins.csv."""
    users_df = _load_users()
    login_counts = df["Username"].value_counts() if not df.empty else pd.Series()
    blocked_counts = (
        df[df["Outcome"] == 1]["Username"].value_counts()
        if not df.empty and "Outcome" in df.columns
        else pd.Series()
    )

    if users_df.empty and df.empty:
        return "📭 No users found."

    # Collect all known usernames from both sources
    all_users = set()
    if not users_df.empty and "Username" in users_df.columns:
        all_users.update(users_df["Username"].dropna().tolist())
    if not df.empty:
        all_users.update(df["Username"].dropna().tolist())

    all_users = sorted(all_users)

    lines = [f"👥 **All Users — {len(all_users)} total:**\n"]
    lines.append("| # | Username | Total Logins | Blocked | Registered At |")
    lines.append("|---|----------|-------------|---------|----------------|")

    # Build lookup for registration date
    reg_map = {}
    if not users_df.empty and "Username" in users_df.columns:
        for _, row in users_df.iterrows():
            uname = row.get("Username", "")
            if not uname or pd.isna(uname):
                continue
            # Try multiple timestamp columns
            reg_ts = (
                row.get("Registration Timestamp")
                or row.get("Registered At")
                or row.get("Timestamp")
                or ""
            )
            if pd.notna(reg_ts) and str(reg_ts).strip():
                try:
                    reg_map[uname] = pd.Timestamp(reg_ts).strftime("%d %b %Y")
                except Exception:
                    reg_map[uname] = str(reg_ts)[:10]
            else:
                reg_map[uname] = "—"

    for i, uname in enumerate(all_users, 1):
        total = login_counts.get(uname, 0)
        blocked = blocked_counts.get(uname, 0)
        reg = reg_map.get(uname, "—")
        blocked_str = f"❌ {blocked}" if blocked > 0 else "✅ 0"
        lines.append(f"| {i} | **{uname}** | {total} | {blocked_str} | {reg} |")

    return "\n".join(lines)


def _intent_today(df: pd.DataFrame, username: str | None) -> str:
    if df.empty:
        return "📭 No login records found yet."
    today = datetime.now().date()
    sub = df if not username else df[df["Username"] == username]
    today_df = (
        sub[sub["Timestamp"].dt.date == today]
        if "Timestamp" in sub.columns
        else pd.DataFrame()
    )
    label = f" for **{username}**" if username else ""
    if today_df.empty:
        return f"📭 No login attempts today{label}."

    lines = [f"📅 **Today's logins{label} ({len(today_df)} total):**\n"]
    lines.append("| Time | User | Status | IP |")
    lines.append("|------|------|--------|----|")
    for _, row in today_df.sort_values("Timestamp", ascending=False).iterrows():
        ts = _fmt_ts(row.get("Timestamp", ""))
        user = row.get("Username", "—")
        outcome = _outcome_label(row.get("Outcome"))
        ip = row.get("IP", "—")
        lines.append(f"| {ts} | **{user}** | {outcome} | `{ip}` |")
    return "\n".join(lines)


def _intent_help() -> str:
    return (
        "🤖 **AccessGuard Login Chatbot** — things you can ask:\n\n"
        "| Question | Example |\n"
        "|----------|---------|\n"
        "| Summary / stats | *show summary*, *how many logins?* |\n"
        "| Recent logins | *last 5 logins*, *recent logins for harsh* |\n"
        "| Blocked attempts | *show blocked logins*, *who was blocked?* |\n"
        "| Last login | *last login for tanvi* |\n"
        "| Devices used | *what devices did harsh use?* |\n"
        "| IPs used | *IP addresses for geet12* |\n"
        "| Risk scores | *risk score stats*, *high risk logins* |\n"
        "| Today's logins | *logins today* |\n"
        "| List all users | *show all users*, *list users* |\n\n"
        "💡 Add a **username** to most questions for user-specific data!"
    )


# ─────────────────────────────────────────────
# Intent classification
# ─────────────────────────────────────────────


def _match(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text, re.IGNORECASE))


def chatbot_response(user_message: str) -> str:
    msg = user_message.strip()
    df = _load_logins()
    user = _extract_username(msg, df)

    # ── Greetings ──
    if _match(r"\b(hi|hello|hey|howdy|hola)\b", msg):
        return (
            "👋 Hello! I'm the **AccessGuard Login Bot**.\n\n"
            "Ask me about login history, blocked attempts, risk scores, and more!\n"
            "Type **help** to see all questions I can answer."
        )

    # ── Help ──
    if _match(r"\b(help|what can you do|commands|options)\b", msg):
        return _intent_help()

    # ── Today ──
    if _match(r"\btoday\b", msg):
        return _intent_today(df, user)

    # ── Blocked attempts ──
    if _match(r"\b(block|blocked|denied|rejected|failed|failure)\b", msg):
        return _intent_blocked(df, user)

    # ── Last login ──
    if _match(r"\blast\s+login\b", msg) or _match(r"\bmost\s+recent\s+login\b", msg):
        return _intent_last_login(df, user)

    # ── Recent N logins — NOTE: "login" OR "logins" (plurals!) ──
    if _match(r"\b(recent|last|latest)\b", msg) and _match(
        r"\blogins?\b|\battempts?\b|\baccess\b", msg
    ):
        n = _extract_n(msg, default=5)
        return _intent_recent(df, user, n)

    # ── Devices ──
    if _match(r"\b(device|os|machine|computer|phone|mobile)\b", msg):
        return _intent_devices(df, user)

    # ── IPs ──
    if _match(r"\b(ip|address|location|where)\b", msg):
        return _intent_ip(df, user)

    # ── Risk score ──
    if _match(r"\b(risk|score|threat|danger|suspicious|high.?risk)\b", msg):
        return _intent_risk(df, user)

    # ── All users ──
    if _match(
        r"\b(users?|all\s+users?|who|list\s+users?|people|everyone|members?)\b", msg
    ):
        return _intent_users(df)

    # ── Summary / stats ──
    if _match(
        r"\b(summary|stats|statistics|total|count|how\s+many|history|logins?|attempts?|report|overview)\b",
        msg,
    ):
        return _intent_summary(df, user)

    # ── Username mentioned but intent unclear → summary ──
    if user:
        return _intent_summary(df, user)

    # ── Fallback ──
    return (
        "🤔 I'm not sure what you mean. Try:\n"
        "- *show blocked attempts*\n"
        "- *last 5 logins*\n"
        "- *show all users*\n"
        "- *risk score stats*\n\n"
        "Or type **help** for all options."
    )
