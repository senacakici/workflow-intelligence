"""
Weak Supervision Pipeline
-------------------------
Pre-labels unlabeled workflow data using heuristic labeling functions.
Inspired by Snorkel's programmatic labeling approach.
No external dependency on snorkel — pure numpy/pandas implementation.
"""

import pandas as pd
import numpy as np
from typing import Callable, List

# Label constants
DEVELOPMENT = 0
REVIEW      = 1
MEETING     = 2
ADMIN       = 3
PLANNING    = 4
ABSTAIN     = -1

LABEL_MAP = {0: "development", 1: "review", 2: "meeting", 3: "admin", 4: "planning"}
REVERSE_MAP = {v: k for k, v in LABEL_MAP.items()}


# ── Labeling Functions ────────────────────────────────────────────────────────

def lf_meeting_keywords(row) -> int:
    keywords = ["standup", "meeting", "sync", "1:1", "all-hands", "demo",
                "retrospective", "interview", "call", "onboarding call"]
    desc = row["task_description"].lower()
    return MEETING if any(k in desc for k in keywords) else ABSTAIN


def lf_development_keywords(row) -> int:
    keywords = ["implement", "build", "code", "refactor", "deploy", "fix bug",
                "unit test", "endpoint", "migrate", "integrate", "sql", "api"]
    desc = row["task_description"].lower()
    return DEVELOPMENT if any(k in desc for k in keywords) else ABSTAIN


def lf_review_keywords(row) -> int:
    keywords = ["review", "audit", "evaluate", "peer review", "pull request", "pr #"]
    desc = row["task_description"].lower()
    return REVIEW if any(k in desc for k in keywords) else ABSTAIN


def lf_admin_keywords(row) -> int:
    keywords = ["expense", "hr", "jira", "timesheet", "paperwork",
                "schedule", "organize", "fill out", "survey"]
    desc = row["task_description"].lower()
    return ADMIN if any(k in desc for k in keywords) else ABSTAIN


def lf_planning_keywords(row) -> int:
    keywords = ["roadmap", "sprint", "research", "gantt", "specification",
                "strategy", "timeline", "outline", "plan", "draft"]
    desc = row["task_description"].lower()
    return PLANNING if any(k in desc for k in keywords) else ABSTAIN


def lf_short_duration_is_admin(row) -> int:
    return ADMIN if row["duration_min"] < 20 else ABSTAIN


def lf_long_meeting(row) -> int:
    desc = row["task_description"].lower()
    if row["duration_min"] > 90 and any(k in desc for k in ["meeting", "sync", "call"]):
        return MEETING
    return ABSTAIN


def lf_morning_planning(row) -> int:
    if row["hour"] <= 9:
        desc = row["task_description"].lower()
        if any(k in desc for k in ["plan", "roadmap", "sprint", "goal"]):
            return PLANNING
    return ABSTAIN


LABELING_FUNCTIONS: List[Callable] = [
    lf_meeting_keywords,
    lf_development_keywords,
    lf_review_keywords,
    lf_admin_keywords,
    lf_planning_keywords,
    lf_short_duration_is_admin,
    lf_long_meeting,
    lf_morning_planning,
]


# ── Label Matrix & Aggregation ────────────────────────────────────────────────

def build_label_matrix(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    m = len(LABELING_FUNCTIONS)
    L = np.full((n, m), ABSTAIN, dtype=int)
    for j, lf in enumerate(LABELING_FUNCTIONS):
        for i, (_, row) in enumerate(df.iterrows()):
            L[i, j] = lf(row)
    return L


def majority_vote(L: np.ndarray) -> np.ndarray:
    """Aggregate label matrix via majority vote (ties → ABSTAIN)."""
    n = L.shape[0]
    predictions = np.full(n, ABSTAIN, dtype=int)
    for i in range(n):
        votes = L[i][L[i] != ABSTAIN]
        if len(votes) == 0:
            continue
        counts = np.bincount(votes, minlength=5)
        top = np.argmax(counts)
        # Require at least 2 agreeing functions OR single clear vote
        if counts[top] >= 1:
            predictions[i] = top
    return predictions


def coverage(L: np.ndarray) -> float:
    """Fraction of rows where at least one LF fires."""
    return (L != ABSTAIN).any(axis=1).mean()


def compute_lf_stats(L: np.ndarray) -> pd.DataFrame:
    stats = []
    for j, lf in enumerate(LABELING_FUNCTIONS):
        col = L[:, j]
        fired = col != ABSTAIN
        stats.append({
            "labeling_function": lf.__name__,
            "coverage": fired.mean().round(4),
            "n_labeled": fired.sum(),
            "label_distribution": {LABEL_MAP[k]: int((col == k).sum())
                                    for k in LABEL_MAP if (col == k).sum() > 0},
        })
    return pd.DataFrame(stats)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv("data/activities.csv")
    unlabeled = df[df["category"].isna()].copy().reset_index(drop=True)

    print(f"Unlabeled records: {len(unlabeled)}")

    L = build_label_matrix(unlabeled)
    preds = majority_vote(L)

    cov = coverage(L)
    labeled_count = (preds != ABSTAIN).sum()

    print(f"\n📊 Weak Supervision Results")
    print(f"   Coverage:         {cov:.1%} of unlabeled rows had ≥1 LF fire")
    print(f"   Pre-labeled:      {labeled_count} / {len(unlabeled)} records")
    print(f"   Still ABSTAIN:    {(preds == ABSTAIN).sum()} records")

    print(f"\n📋 Labeling Function Stats:")
    stats_df = compute_lf_stats(L)
    print(stats_df[["labeling_function", "coverage", "n_labeled"]].to_string(index=False))

    # Attach predictions back
    unlabeled["predicted_category"] = [
        LABEL_MAP[p] if p != ABSTAIN else None for p in preds
    ]

    # Merge back with labeled data
    labeled = df[df["category"].notna()].copy()
    labeled["predicted_category"] = labeled["category"]

    final_df = pd.concat([labeled, unlabeled], ignore_index=True)
    final_df.to_csv("data/activities_prelabeled.csv", index=False)

    print(f"\n✅ Saved prelabeled dataset → data/activities_prelabeled.csv")
    print(f"   Total rows now labeled: {final_df['predicted_category'].notna().sum()} / {len(final_df)}")
