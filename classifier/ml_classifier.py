"""
classifier/ml_classifier.py
─────────────────────────────
Machine learning classifier trained on labeled lobster neural recordings.

Model: RandomForestClassifier (sklearn)
  - Robust to small datasets
  - Outputs calibrated probability scores
  - Feature importance available for debugging

Training workflow:
  1. Run `scripts/record_baseline.py` to capture labeled recordings
  2. Run `scripts/train_classifier.py` to train and save the model
  3. Set classifier.mode = "ml" in config.yaml

Feature vector (6 dimensions):
  [spike_rate, isi_mean_ms, isi_cv, burst_score, amplitude_mean, amplitude_std]
"""

import logging
import numpy as np
import joblib
import os
from typing import Optional
from ..processing.features import NeuralFeatures
from .threshold import ClassifierResult, Signal

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "HOLD", 1: "LONG", 2: "SHORT"}
REVERSE_LABEL_MAP = {"HOLD": 0, "LONG": 1, "SHORT": 2}


class MLClassifier:
    """
    Scikit-learn based classifier for neural trading signals.
    Falls back to HOLD if model is not loaded or confidence is below threshold.
    """

    def __init__(self, model_path: str, min_confidence: float):
        self.model_path = model_path
        self.min_confidence = min_confidence
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        if os.path.exists(self.model_path):
            try:
                self._model = joblib.load(self.model_path)
                logger.info(f"ML classifier loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load ML model from {self.model_path}: {e}")
                self._model = None
        else:
            logger.warning(
                f"ML model not found at {self.model_path}. "
                "Run scripts/train_classifier.py first, or use threshold mode."
            )

    def classify(self, features: NeuralFeatures) -> ClassifierResult:
        """Classify features using the trained model."""
        if self._model is None:
            return ClassifierResult(
                signal="HOLD",
                confidence=0.0,
                spike_rate=features.spike_rate,
                burst_score=features.burst_score,
                reason="ML model not loaded — defaulting to HOLD",
            )

        X = features.to_array().reshape(1, -1)

        try:
            proba = self._model.predict_proba(X)[0]   # shape: (n_classes,)
            class_idx = int(np.argmax(proba))
            confidence = float(proba[class_idx])
            signal: Signal = LABEL_MAP.get(class_idx, "HOLD")

            if confidence < self.min_confidence:
                return ClassifierResult(
                    signal="HOLD",
                    confidence=confidence,
                    spike_rate=features.spike_rate,
                    burst_score=features.burst_score,
                    reason=f"ML confidence {confidence:.2f} below threshold {self.min_confidence}",
                )

            return ClassifierResult(
                signal=signal,
                confidence=confidence,
                spike_rate=features.spike_rate,
                burst_score=features.burst_score,
                reason=f"ML: {signal} @ {confidence:.2f} confidence",
            )

        except Exception as e:
            logger.error(f"ML classification error: {e}")
            return ClassifierResult(
                signal="HOLD",
                confidence=0.0,
                spike_rate=features.spike_rate,
                burst_score=features.burst_score,
                reason=f"ML error: {e}",
            )

    def is_loaded(self) -> bool:
        return self._model is not None

    def feature_importances(self) -> Optional[dict]:
        """Return feature importance dict if model supports it."""
        if self._model is None:
            return None
        if hasattr(self._model, "feature_importances_"):
            names = [
                "spike_rate", "isi_mean_ms", "isi_cv",
                "burst_score", "amplitude_mean", "amplitude_std"
            ]
            return dict(zip(names, self._model.feature_importances_.tolist()))
        return None
