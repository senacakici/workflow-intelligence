import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

TEAMS = ["engineering", "product", "design", "marketing", "ops"]
CATEGORIES = ["development", "review", "meeting", "admin", "planning"]

TASK_TEMPLATES = {
    "development": [
        "Implement authentication module",
        "Fix bug in payment flow",
        "Refactor database queries",
        "Write unit tests for API",
        "Code review for PR #{}",
        "Deploy microservice to staging",
        "Optimize SQL queries for dashboard",
        "Build REST endpoint for user profile",
        "Migrate legacy codebase to Python 3",
        "Integrate third-party analytics SDK",
    ],
    "review": [
        "Review pull request for feature branch",
        "Conduct design review with team",
        "Review quarterly OKR progress",
        "Peer review for documentation",
        "Code audit for security vulnerabilities",
        "Review test coverage report",
        "Evaluate vendor proposal",
        "Review analytics dashboard mockup",
    ],
    "meeting": [
        "Weekly team standup",
        "Sprint planning meeting",
        "1:1 with manager",
        "All-hands company meeting",
        "Cross-team sync on roadmap",
        "Client onboarding call",
        "Retrospective meeting",
        "Product demo to stakeholders",
        "Interview candidate for senior role",
        "Incident postmortem discussion",
    ],
    "admin": [
        "Update project documentation",
        "Fill out expense report",
        "Respond to HR policy survey",
        "Update Jira tickets",
        "Schedule quarterly performance review",
        "Organize shared drive folders",
        "Submit timesheet for approval",
        "Onboarding paperwork for new hire",
    ],
    "planning": [
        "Draft Q3 product roadmap",
        "Define sprint goals and priorities",
        "Research competitors for strategy session",
        "Create project timeline in Gantt chart",
        "Write technical specification document",
        "Plan infrastructure scaling strategy",
        "Map out user journey for new feature",
        "Outline go-to-market plan",
    ],
}

DURATION_PARAMS = {
    "development": (90, 40),
    "review":      (45, 20),
    "meeting":     (55, 25),
    "admin":       (30, 15),
    "planning":    (60, 30),
}


def generate_users(n=25):
    users = []
    for i in range(1, n + 1):
        users.append({
            "user_id": f"U{i:03d}",
            "team": random.choice(TEAMS),
            "seniority": random.choice(["junior", "mid", "senior"]),
        })
    return pd.DataFrame(users)


def simulate_activities(users_df, n_records=1200):
    records = []
    start_date = datetime(2024, 1, 1)

    for _ in range(n_records):
        user = users_df.sample(1).iloc[0]

        # Weighted category distribution per team
        if user["team"] == "engineering":
            weights = [0.45, 0.20, 0.15, 0.10, 0.10]
        elif user["team"] == "product":
            weights = [0.10, 0.20, 0.20, 0.10, 0.40]
        elif user["team"] == "design":
            weights = [0.20, 0.30, 0.20, 0.10, 0.20]
        else:
            weights = [0.10, 0.15, 0.30, 0.25, 0.20]

        category = random.choices(CATEGORIES, weights=weights)[0]
        template = random.choice(TASK_TEMPLATES[category])
        if "{}" in template:
            template = template.format(random.randint(100, 999))

        mu, sigma = DURATION_PARAMS[category]
        duration = max(5, int(np.random.normal(mu, sigma)))

        # Random timestamp within business hours
        days_offset = random.randint(0, 180)
        hour = random.randint(8, 17)
        minute = random.choice([0, 15, 30, 45])
        timestamp = start_date + timedelta(days=days_offset, hours=hour, minutes=minute)

        # Skip weekends
        if timestamp.weekday() >= 5:
            timestamp += timedelta(days=2)

        # Inject anomalies (~5% of records)
        is_anomaly = random.random() < 0.05
        if is_anomaly:
            duration = duration * random.randint(3, 6)

        # Leave 25% unlabeled (for weak supervision demo)
        label = category if random.random() > 0.25 else None

        records.append({
            "user_id": user["user_id"],
            "team": user["team"],
            "seniority": user["seniority"],
            "task_description": template,
            "category": label,
            "duration_min": duration,
            "timestamp": timestamp,
            "is_anomaly": is_anomaly,
            "day_of_week": timestamp.strftime("%A"),
            "hour": timestamp.hour,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    users = generate_users(25)
    activities = simulate_activities(users, 1200)

    users.to_csv("data/users.csv", index=False)
    activities.to_csv("data/activities.csv", index=False)

    print(f"✅ Generated {len(activities)} activity records")
    print(f"   Labeled:   {activities['category'].notna().sum()}")
    print(f"   Unlabeled: {activities['category'].isna().sum()}")
    print(f"   Anomalies: {activities['is_anomaly'].sum()}")
    print(f"\nCategory distribution (labeled):")
    print(activities['category'].value_counts())
