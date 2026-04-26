# Workflow Intelligence Engine

I built this to understand how workforce analytics platforms work under the hood — specifically the part where raw activity logs get turned into something useful without requiring someone to manually label thousands of rows.

The dataset is simulated (1200 employee activity records across 5 teams), but the pipeline is real: text classification, weak supervision for pre-labeling, anomaly detection for workload spikes, a FastAPI endpoint, and a Streamlit dashboard to tie it together.

---

## Architecture

```
Raw Activity Data
       │
       ▼
┌─────────────────┐     ┌──────────────────────────┐
│  Data Simulator │────▶│  activities.csv           │
│  (1200 records) │     │  25% unlabeled            │
└─────────────────┘     └────────────┬─────────────┘
                                     │
              ┌──────────────────────┼────────────────────────┐
              ▼                      ▼                        ▼
   ┌──────────────────┐  ┌───────────────────────┐  ┌────────────────────┐
   │  Classification  │  │   Weak Supervision    │  │ Anomaly Detection  │
   │  TF-IDF + LR     │  │   8 Labeling Functions│  │ Isolation Forest   │
   │  F1 = 1.00       │  │   94.8% coverage      │  │ + Z-Score Analysis │
   └────────┬─────────┘  └──────────┬────────────┘  └────────┬───────────┘
            │                       │                         │
            └───────────────────────┼─────────────────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │   FastAPI Endpoint  │
                         │   /predict          │
                         │   /batch-predict    │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  Streamlit Dashboard │
                         │  Live classification │
                         │  Anomaly alerts      │
                         └─────────────────────┘
```

---

## Results

| Metric | Value |
|--------|-------|
| Classification F1 (weighted) | **1.00** |
| Weak supervision coverage | **94.8%** |
| Records pre-labeled (of 305 unlabeled) | **289 / 305** |
| Anomalous sessions detected | **60** |
| Overload alerts generated | **29 user-weeks** |

---

## Project Structure

```
workflow-intelligence/
│
├── data/
│   ├── simulate_workflow_data.py    # Generates realistic activity data
│   ├── activities.csv               # Raw simulated data (25% unlabeled)
│   ├── activities_prelabeled.csv    # After weak supervision
│   ├── activities_scored.csv        # After anomaly scoring
│   ├── workload_alerts.csv          # Anomaly alerts per user
│   └── weekly_workload.csv          # Weekly aggregates
│
├── models/
│   ├── task_classifier.py           # TF-IDF + Logistic Regression pipeline
│   ├── weak_supervision.py          # 8 labeling functions + majority vote
│   ├── anomaly_detection.py         # Isolation Forest + Z-score
│   └── task_classifier.pkl          # Serialized best model
│
├── api/
│   └── predict.py                   # FastAPI /predict endpoint
│
├── dashboard/
│   └── app.py                       # Streamlit interactive dashboard
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/workflow-intelligence
cd workflow-intelligence
pip install -r requirements.txt

# 1. Generate data
python data/simulate_workflow_data.py

# 2. Train classifier
python models/task_classifier.py

# 3. Run weak supervision
python models/weak_supervision.py

# 4. Run anomaly detection
python models/anomaly_detection.py

# 5. Start API
uvicorn api.predict:app --reload

# 6. Start Dashboard
streamlit run dashboard/app.py
```

---

## API

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"description": "Weekly team standup with engineering"}'

# Response
{
  "category": "meeting",
  "confidence": 0.9821,
  "all_probabilities": {
    "admin": 0.003,
    "development": 0.002,
    "meeting": 0.982,
    "planning": 0.008,
    "review": 0.005
  }
}
```

---

## Weak Supervision

305 of the 1200 records had no label. Instead of labeling them manually, I wrote 8 labeling functions that apply simple keyword and duration-based rules. Each function either assigns a label or abstains. Final label is decided by majority vote across all functions.

```python
def lf_meeting_keywords(row):
    keywords = ["standup", "meeting", "sync", "1:1", "all-hands"]
    return MEETING if any(k in row["task_description"].lower() for k in keywords) else ABSTAIN
```

Coverage across all 8 functions:

| Labeling Function | Coverage | Records Labeled |
|-------------------|----------|-----------------|
| lf_meeting_keywords | 23.3% | 71 |
| lf_planning_keywords | 24.3% | 74 |
| lf_development_keywords | 21.0% | 64 |
| lf_review_keywords | 22.0% | 67 |
| lf_admin_keywords | 15.7% | 48 |
| lf_short_duration_is_admin | 12.1% | 37 |
| lf_long_meeting | 2.3% | 7 |
| lf_morning_planning | 2.3% | 7 |

Combined: 289 of 305 unlabeled records got a label (94.8% coverage).

---

## Stack

- Python 3.11
- pandas, numpy
- scikit-learn (TF-IDF, Logistic Regression, Isolation Forest)
- FastAPI + uvicorn
- Streamlit
- Matplotlib

---

## Contact

[Your Name] · [LinkedIn] · [GitHub]
