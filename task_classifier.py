import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
import pickle
import warnings
warnings.filterwarnings("ignore")


def load_labeled_data(path="data/activities.csv"):
    df = pd.read_csv(path)
    labeled = df[df["category"].notna()].copy()
    return labeled


def build_pipeline(classifier="lr"):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
    )
    if classifier == "lr":
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    else:
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train_and_evaluate():
    df = load_labeled_data()
    X = df["task_description"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}
    for name, clf_type in [("Logistic Regression", "lr"), ("Random Forest", "rf")]:
        pipe = build_pipeline(clf_type)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="weighted")
        results[name] = {"pipeline": pipe, "f1": f1, "y_pred": y_pred}
        print(f"\n{'='*50}")
        print(f"  {name}  —  Weighted F1: {f1:.4f}")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred))

    # Save best model
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_pipe = results[best_name]["pipeline"]
    with open("models/task_classifier.pkl", "wb") as f:
        pickle.dump(best_pipe, f)
    print(f"\n✅ Best model saved: {best_name} (F1={results[best_name]['f1']:.4f})")

    return best_pipe, X_test, y_test


def predict(text: str, model_path="models/task_classifier.pkl"):
    with open(model_path, "rb") as f:
        pipe = pickle.load(f)
    pred = pipe.predict([text])[0]
    proba = pipe.predict_proba([text])[0]
    confidence = max(proba)
    return {"category": pred, "confidence": round(float(confidence), 4)}


if __name__ == "__main__":
    train_and_evaluate()
