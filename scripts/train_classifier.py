"""
scripts/train_classifier.py
─────────────────────────────
Trains the RandomForest ML classifier on labeled recordings.

Usage:
  python scripts/train_classifier.py

Reads all CSVs from data/recordings/, trains the model,
evaluates with cross-validation, and saves to classifier/models/lobster_model.pkl
"""

import os
import sys
import glob
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURE_COLS = [
    "spike_rate", "isi_mean_ms", "isi_cv",
    "burst_score", "amplitude_mean", "amplitude_std"
]
MODEL_PATH = "classifier/models/lobster_model.pkl"
LABEL_PATH = "classifier/models/label_encoder.pkl"


def main():
    # ── Load all recordings ────────────────────────────────────────────────────
    csv_files = glob.glob("data/recordings/*.csv")
    if not csv_files:
        print("❌ No recordings found in data/recordings/")
        print("   Run scripts/record_baseline.py first.")
        sys.exit(1)

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        print(f"  Loaded {len(df):4d} samples from {os.path.basename(f)}")
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data = data.dropna()

    print(f"\n📊 Total samples: {len(data)}")
    print(data["label"].value_counts().to_string())

    X = data[FEATURE_COLS].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(data["label"].values)

    print(f"\nLabel mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ── Train ──────────────────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"\n📈 Cross-validation accuracy: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

    # Final fit on all data
    model.fit(X, y)
    y_pred = model.predict(X)
    print("\nClassification Report (train set):")
    print(classification_report(y, y_pred, target_names=le.classes_))

    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    print("\nFeature Importances:")
    for feat, imp in sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: -x[1]
    ):
        bar = "█" * int(imp * 50)
        print(f"  {feat:20s}  {imp:.4f}  {bar}")

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs("classifier/models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, LABEL_PATH)

    metadata = {
        "n_samples": len(data),
        "features": FEATURE_COLS,
        "classes": le.classes_.tolist(),
        "cv_accuracy_mean": float(scores.mean()),
        "cv_accuracy_std": float(scores.std()),
        "n_estimators": model.n_estimators,
    }
    with open("classifier/models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Model saved to {MODEL_PATH}")
    print(f"✅ Label encoder saved to {LABEL_PATH}")
    print("\nTo use the ML classifier, set in config.yaml:")
    print('  classifier:\n    mode: "ml"')


if __name__ == "__main__":
    main()
