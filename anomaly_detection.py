"""
Anomaly Detection — Workforce Intelligence
------------------------------------------
Detects abnormal workload patterns using:
  1. Isolation Forest on session-level features
  2. Z-score analysis on user weekly workload
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


def load_data(path="data/activities_prelabeled.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["month"] = df["timestamp"].dt.month
    return df


def isolation_forest_detection(df: pd.DataFrame):
    """Flag anomalous individual activity records."""
    features = df[["duration_min", "hour"]].copy()

    # Encode categorical
    le = LabelEncoder()
    features["team_enc"] = le.fit_transform(df["team"])

    clf = IsolationForest(contamination=0.05, random_state=42)
    preds = clf.fit_predict(features)
    scores = clf.decision_function(features)

    df = df.copy()
    df["if_anomaly"] = (preds == -1)
    df["anomaly_score"] = scores
    return df


def zscore_workload_analysis(df: pd.DataFrame):
    """Flag users with abnormal weekly workload."""
    weekly = (
        df.groupby(["user_id", "team", "week"])["duration_min"]
        .sum()
        .reset_index()
        .rename(columns={"duration_min": "weekly_minutes"})
    )

    # Z-score per user across their weeks
    stats = weekly.groupby("user_id")["weekly_minutes"].agg(["mean", "std"]).reset_index()
    stats.columns = ["user_id", "mean_weekly", "std_weekly"]
    stats["std_weekly"] = stats["std_weekly"].fillna(1)

    weekly = weekly.merge(stats, on="user_id")
    weekly["z_score"] = (weekly["weekly_minutes"] - weekly["mean_weekly"]) / weekly["std_weekly"]
    weekly["workload_anomaly"] = weekly["z_score"].abs() > 2.0

    return weekly


def generate_alerts(weekly_df: pd.DataFrame) -> pd.DataFrame:
    alerts = weekly_df[weekly_df["workload_anomaly"]].copy()
    alerts["alert_type"] = alerts["z_score"].apply(
        lambda z: "🔴 OVERLOAD" if z > 2 else "🟡 UNDERLOAD"
    )
    alerts["description"] = alerts.apply(
        lambda r: (
            f"User {r['user_id']} had {r['weekly_minutes']:.0f} min in week {r['week']} "
            f"(avg: {r['mean_weekly']:.0f} min, z={r['z_score']:.2f})"
        ),
        axis=1,
    )
    return alerts[["user_id", "week", "weekly_minutes", "z_score", "alert_type", "description"]]


if __name__ == "__main__":
    df = load_data()

    # Isolation Forest
    df_scored = isolation_forest_detection(df)
    if_count = df_scored["if_anomaly"].sum()
    precision = (df_scored[df_scored["if_anomaly"]]["is_anomaly"]).mean()
    print(f"🔍 Isolation Forest")
    print(f"   Flagged:   {if_count} records as anomalous")
    print(f"   Precision vs ground truth: {precision:.1%}\n")

    # Z-score workload
    weekly = zscore_workload_analysis(df)
    alerts = generate_alerts(weekly)
    print(f"📊 Z-Score Workload Analysis")
    print(f"   Anomalous user-weeks: {len(alerts)}")
    print(f"   Overload alerts:  {(alerts['alert_type'].str.contains('OVER')).sum()}")
    print(f"   Underload alerts: {(alerts['alert_type'].str.contains('UNDER')).sum()}\n")
    print("Top alerts:")
    print(alerts.head(8).to_string(index=False))

    # Save
    df_scored.to_csv("data/activities_scored.csv", index=False)
    alerts.to_csv("data/workload_alerts.csv", index=False)
    weekly.to_csv("data/weekly_workload.csv", index=False)
    print("\n✅ Saved: activities_scored.csv, workload_alerts.csv, weekly_workload.csv")
