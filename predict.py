"""
Workflow Intelligence API
-------------------------
Run with: uvicorn api.predict:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from pathlib import Path

app = FastAPI(
    title="Workflow Intelligence API",
    description="Pre-label and classify workflow tasks automatically",
    version="1.0.0",
)

MODEL_PATH = Path("models/task_classifier.pkl")
model = None


def load_model():
    global model
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        raise FileNotFoundError("Model not found. Run models/task_classifier.py first.")


@app.on_event("startup")
def startup_event():
    load_model()


class TaskInput(BaseModel):
    description: str

    class Config:
        json_schema_extra = {
            "example": {"description": "Weekly team standup with engineering"}
        }


class PredictionOutput(BaseModel):
    category: str
    confidence: float
    all_probabilities: dict


@app.get("/")
def root():
    return {"status": "ok", "message": "Workflow Intelligence API is running"}


@app.post("/predict", response_model=PredictionOutput)
def predict_task(task: TaskInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not task.description.strip():
        raise HTTPException(status_code=400, detail="Description cannot be empty")

    proba = model.predict_proba([task.description])[0]
    classes = model.classes_
    predicted_idx = np.argmax(proba)

    return PredictionOutput(
        category=classes[predicted_idx],
        confidence=round(float(proba[predicted_idx]), 4),
        all_probabilities={
            cls: round(float(p), 4) for cls, p in zip(classes, proba)
        },
    )


@app.post("/batch-predict")
def batch_predict(tasks: list[TaskInput]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    descriptions = [t.description for t in tasks]
    preds = model.predict(descriptions)
    probas = model.predict_proba(descriptions)
    return [
        {"description": d, "category": p, "confidence": round(float(max(prob)), 4)}
        for d, p, prob in zip(descriptions, preds, probas)
    ]


@app.get("/health")
def health():
    return {"model_loaded": model is not None, "status": "healthy"}
