import numpy as np
import pandas as pd


def calibrate(scores, target_fpr=0.01):

    scores = np.asarray(scores, dtype=float)

    scores = scores[np.isfinite(scores)]

    if len(scores) == 0:
        raise ValueError("No valid scores")

    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be between 0 and 1")

    threshold = np.quantile(
        scores,
        1 - target_fpr,
        method="higher"
    )

    return threshold
def evaluate(alerts, test_df):

    alerts = np.asarray(alerts, dtype=bool)

    df = test_df.reset_index(drop=True).copy()

    if len(alerts) != len(df):
        raise ValueError("Alerts and dataframe must have same length")

    is_attack = pd.to_numeric(
        df["is_attack"],
        errors="coerce"
    ).fillna(0).to_numpy() == 1

    benign_event = (
        df["benign_event"]
        .notna()
        & ~df["benign_event"].astype(str).str.strip().str.lower().isin(
            ["", "none", "nan", "null"]
        )
    ).to_numpy()

    # Normal rows: no attack and no benign event
    normal_mask = (~is_attack) & (~benign_event)

    # False Positive Rate
    fpr = (
        alerts[normal_mask].mean()
        if normal_mask.sum() > 0
        else np.nan
    )

    # Alert rate on benign events
    benign_alert_rate = (
        alerts[benign_event].mean()
        if benign_event.sum() > 0
        else np.nan
    )

    # Recall by attack type
    recall_per_type = {}

    attack_types = df.loc[is_attack, "attack_type"].dropna().unique()

    for attack_type in attack_types:

        mask = (
            is_attack
            & (df["attack_type"].to_numpy() == attack_type)
        )

        recall_per_type[attack_type] = (
            alerts[mask].mean()
        )

    # Total alerts
    total_alerts = int(alerts.sum())

    # Alerts per day
    timestamps = pd.to_datetime(df["timestamp"])

    number_of_days = timestamps.dt.date.nunique()

    alerts_per_day = (
        total_alerts / number_of_days
        if number_of_days > 0
        else 0
    )

    return {
        "FPR": fpr,
        "Benign Alert Rate": benign_alert_rate,
        "Total Alerts": total_alerts,
        "Alerts Per Day": alerts_per_day,
        "Recall Per Type": recall_per_type
    }
def precision_at_k(scores, test_df, k=100):

    scores = np.asarray(scores, dtype=float)

    is_attack = (
        pd.to_numeric(
            test_df["is_attack"],
            errors="coerce"
        ).fillna(0).to_numpy() == 1
    )

    if len(scores) != len(is_attack):
        raise ValueError("Scores and test_df must have same length")

    if not np.isfinite(scores).all():
        raise ValueError("Scores contain invalid values")

    k = min(k, len(scores))

    if k == 0:
        return np.nan

    top_indices = np.argsort(scores)[::-1][:k]

    return is_attack[top_indices].mean()