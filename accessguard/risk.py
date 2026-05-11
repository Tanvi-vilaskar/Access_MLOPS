# risk.py

from utils import get_geolocation, login_attempts_in_last_hour
from profile_model import get_user_profile

# Tunable thresholds
GEO_MISMATCH_SCORE = 25
TIME_DEVIATION_SCORE = 20
ANOMALY_COUNT_BONUS = 25


def _hour_distance(h1, h2):
    # circular distance between hours (0-23)
    d = abs(h1 - h2)
    return min(d, 24 - d)


def predict_login(
    username,
    ip,
    device,
    browser,
    hour,
    mfa_enabled,
    registered_ip=None,
    registered_device=None,
    registered_browser=None,
):
    """
    Enhanced risk scoring:
     - original checks (IP/device/browser)
     - geolocation mismatch check using IP -> country/city
     - per-user usual hour check (profile) -> penalize if far from usual hour
     - anomaly_count causes additional penalty
    """
    score = 0
    reasons = []

    # Basic comparisons (existing)
    if ip and registered_ip and ip != registered_ip:
        score += 30
        reasons.append(f"Login from new IP: {ip}")

    if device and registered_device and device != registered_device:
        score += 25
        reasons.append(f"Different device detected: {device}")

    if browser and registered_browser and browser != registered_browser:
        score += 15
        reasons.append(f"Different browser detected: {browser}")

    # Geolocation check (country-level)
    try:
        current_loc = get_geolocation(ip)
        registered_loc = None
        # Try to obtain registered geolocation from user profile
        profile = get_user_profile(username)
        registered_loc = profile.get("last_geolocation") if profile else None

        if (
            registered_loc
            and current_loc
            and registered_loc != "Unknown"
            and current_loc != "Unknown"
        ):
            # Compare main country strings (split by comma)
            cur_country = current_loc.split(",")[0].strip()
            reg_country = registered_loc.split(",")[0].strip()
            if cur_country and reg_country and cur_country != reg_country:
                score += GEO_MISMATCH_SCORE
                reasons.append(
                    f"Geolocation mismatch: {current_loc} vs {registered_loc}"
                )
    except Exception:
        # don't fail entirely on geolocation service errors
        pass

    # Time-of-day anomaly using user profile (mean hour + std)
    try:
        if profile:
            mean_hour = profile.get("mean_hour")
            std_hour = profile.get("std_hour", 0)
            if mean_hour is not None:
                dist = _hour_distance(hour, int(round(mean_hour)))
                # penalize proportionally: far > 4 hours => apply full TIME_DEVIATION_SCORE
                if dist >= 4:
                    score += TIME_DEVIATION_SCORE
                    reasons.append(
                        f"Login at unusual hour: {hour} (user average: {mean_hour:.1f} ± {std_hour:.1f})"
                    )
                elif dist > 1:
                    # smaller penalty
                    add = int((dist / 4.0) * TIME_DEVIATION_SCORE)
                    score += add
                    reasons.append(
                        f"Login hour somewhat unusual: {hour} (avg {mean_hour:.1f})"
                    )
    except Exception:
        pass

    # Rapid attempts
    attempts_last_hour = login_attempts_in_last_hour(username)
    if attempts_last_hour >= 5:
        score += 20
        reasons.append(f"{attempts_last_hour} login attempts in last hour")

    # Combined anomaly boost
    anomaly_count = sum(
        [
            (ip and registered_ip and ip != registered_ip),
            (device and registered_device and device != registered_device),
            (browser and registered_browser and browser != registered_browser),
        ]
    )
    if anomaly_count >= 2:
        score += ANOMALY_COUNT_BONUS
        reasons.append("Multiple anomalies detected")

    # Final decision
    if score < 30:
        decision = "ALLOW ✅"
    elif score < 60:
        if mfa_enabled:
            decision = "ALLOW with MFA 🔐"
            reasons.append("MFA enforced due to medium risk")
        else:
            decision = "BLOCK ❌"
            reasons.append("Medium risk but MFA not enabled")
    else:
        decision = "BLOCK ❌"
        reasons.append("High risk login attempt")

    return score, decision, reasons
