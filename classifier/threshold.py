"""
classifier/threshold.py
────────────────────────
Rule-based signal classifier.

Maps neural feature vectors to trading signals using configurable thresholds.

Signal logic:
  - LONG:  high spike rate + high burst score (excitatory/active state)
  - SHORT: low spike rate + low burst score (inhibitory/quiescent state)
  - HOLD:  intermediate state — no action taken

Confidence is computed as the normalized distance from the decision boundary.
"""

import logging
from dataclasses import dataclass
from typing import Literal
from ..processing.features import NeuralFeatures

logger = logging.getLogger(__name__)

Signal = Literal["LONG", "SHORT", "HOLD"]


@dataclass
class ClassifierResult:
    signal: Signal
    confidence: float        # 0.0 – 1.0
    spike_rate: float
    burst_score: float
    reason: str


class ThresholdClassifier:
    """
    Dual-threshold classifier operating on spike rate and burst score.

    Decision boundaries:
      LONG  if spike_rate >= long_rate  AND burst_score >= long_burst
      SHORT if spike_rate <= short_rate AND burst_score <= short_burst
      HOLD  otherwise
    """

    def __init__(
        self,
        long_spike_rate: float,
        short_spike_rate: float,
        long_burst_score: float,
        short_burst_score: float,
        min_confidence: float,
    ):
        self.long_rate = long_spike_rate
        self.short_rate = short_spike_rate
        self.long_burst = long_burst_score
        self.short_burst = short_burst_score
        self.min_confidence = min_confidence

        # Validate thresholds
        if short_spike_rate >= long_spike_rate:
            raise ValueError(
                f"short_spike_rate ({short_spike_rate}) must be < long_spike_rate ({long_spike_rate})"
            )

    def classify(self, features: NeuralFeatures) -> ClassifierResult:
        """
        Classify a feature window into a trading signal.

        Returns ClassifierResult with signal="HOLD" if confidence is below minimum.
        """
        rate = features.spike_rate
        burst = features.burst_score

        # ── LONG conditions ────────────────────────────────────────────────────
        if rate >= self.long_rate and burst >= self.long_burst:
            # Confidence: how far above both thresholds
            rate_conf = min(1.0, (rate - self.long_rate) / self.long_rate)
            burst_conf = min(1.0, (burst - self.long_burst) / (1.0 - self.long_burst + 1e-9))
            confidence = 0.6 * rate_conf + 0.4 * burst_conf
            if confidence >= self.min_confidence:
                return ClassifierResult(
                    signal="LONG",
                    confidence=confidence,
                    spike_rate=rate,
                    burst_score=burst,
                    reason=f"Rate={rate:.1f}Hz≥{self.long_rate}, Burst={burst:.2f}≥{self.long_burst}",
                )

        # ── SHORT conditions ───────────────────────────────────────────────────
        elif rate <= self.short_rate and burst <= self.short_burst:
            rate_conf = min(1.0, (self.short_rate - rate) / (self.short_rate + 1e-9))
            burst_conf = min(1.0, (self.short_burst - burst) / (self.short_burst + 1e-9))
            confidence = 0.6 * rate_conf + 0.4 * burst_conf
            if confidence >= self.min_confidence:
                return ClassifierResult(
                    signal="SHORT",
                    confidence=confidence,
                    spike_rate=rate,
                    burst_score=burst,
                    reason=f"Rate={rate:.1f}Hz≤{self.short_rate}, Burst={burst:.2f}≤{self.short_burst}",
                )

        # ── HOLD ───────────────────────────────────────────────────────────────
        return ClassifierResult(
            signal="HOLD",
            confidence=0.0,
            spike_rate=rate,
            burst_score=burst,
            reason=f"No clear signal: Rate={rate:.1f}Hz, Burst={burst:.2f}",
        )
