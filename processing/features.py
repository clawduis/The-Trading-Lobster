"""
processing/features.py
───────────────────────
Feature extraction from detected spike trains.

Features computed over a sliding time window:
  - spike_rate:     spikes per second (Hz)
  - isi_mean:       mean inter-spike interval (ms)
  - isi_cv:         coefficient of variation of ISI (burstiness measure)
  - burst_score:    fraction of spikes in bursts (ISI < burst_threshold_ms)
  - amplitude_mean: mean spike amplitude
  - amplitude_std:  std dev of spike amplitudes (waveform consistency)

These features are used by both the threshold classifier and ML classifier.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import List
from .spike_detector import Spike

logger = logging.getLogger(__name__)

BURST_ISI_THRESHOLD_MS = 20.0   # ISIs shorter than this = burst


@dataclass
class NeuralFeatures:
    """Feature vector extracted from a window of spike data."""
    window_start: float       # seconds
    window_end: float         # seconds
    spike_count: int
    spike_rate: float         # Hz
    isi_mean_ms: float        # ms
    isi_cv: float             # dimensionless
    burst_score: float        # 0.0 – 1.0
    amplitude_mean: float
    amplitude_std: float

    def to_array(self) -> np.ndarray:
        """Flat feature vector for ML classifier input."""
        return np.array([
            self.spike_rate,
            self.isi_mean_ms,
            self.isi_cv,
            self.burst_score,
            self.amplitude_mean,
            self.amplitude_std,
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "window_start": self.window_start,
            "window_end": self.window_end,
            "spike_count": self.spike_count,
            "spike_rate_hz": self.spike_rate,
            "isi_mean_ms": self.isi_mean_ms,
            "isi_cv": self.isi_cv,
            "burst_score": self.burst_score,
            "amplitude_mean": self.amplitude_mean,
            "amplitude_std": self.amplitude_std,
        }


def extract_features(
    spikes: List[Spike],
    window_start: float,
    window_end: float,
) -> NeuralFeatures:
    """
    Compute feature vector from a list of spikes within [window_start, window_end].

    Args:
        spikes:       All Spike objects from the detector (will be filtered to window)
        window_start: Start of analysis window (seconds)
        window_end:   End of analysis window (seconds)

    Returns:
        NeuralFeatures instance
    """
    window_sec = window_end - window_start
    if window_sec <= 0:
        raise ValueError("window_end must be greater than window_start")

    # Filter spikes to this window
    window_spikes = [s for s in spikes if window_start <= s.timestamp_sec < window_end]
    n = len(window_spikes)

    spike_rate = n / window_sec

    if n < 2:
        return NeuralFeatures(
            window_start=window_start,
            window_end=window_end,
            spike_count=n,
            spike_rate=spike_rate,
            isi_mean_ms=0.0,
            isi_cv=0.0,
            burst_score=0.0,
            amplitude_mean=window_spikes[0].amplitude if n == 1 else 0.0,
            amplitude_std=0.0,
        )

    timestamps = np.array([s.timestamp_sec for s in window_spikes])
    amplitudes = np.array([s.amplitude for s in window_spikes])

    # Inter-spike intervals
    isis_sec = np.diff(timestamps)
    isis_ms = isis_sec * 1000.0

    isi_mean = float(np.mean(isis_ms))
    isi_std = float(np.std(isis_ms))
    isi_cv = isi_std / isi_mean if isi_mean > 0 else 0.0

    # Burst score: fraction of ISIs below burst threshold
    n_burst_isis = int(np.sum(isis_ms < BURST_ISI_THRESHOLD_MS))
    burst_score = n_burst_isis / len(isis_ms) if len(isis_ms) > 0 else 0.0

    return NeuralFeatures(
        window_start=window_start,
        window_end=window_end,
        spike_count=n,
        spike_rate=spike_rate,
        isi_mean_ms=isi_mean,
        isi_cv=isi_cv,
        burst_score=burst_score,
        amplitude_mean=float(np.mean(amplitudes)),
        amplitude_std=float(np.std(amplitudes)),
    )


class SlidingWindowFeatureExtractor:
    """
    Maintains a rolling spike buffer and extracts features on a sliding window.
    """

    def __init__(self, window_size_sec: float, window_step_sec: float):
        self.window_size = window_size_sec
        self.window_step = window_step_sec
        self._spike_buffer: List[Spike] = []
        self._next_window_start: float = 0.0
        self._initialized = False

    def add_spikes(self, spikes: List[Spike]) -> List[NeuralFeatures]:
        """
        Add new spikes and return any completed feature windows.

        Args:
            spikes: Newly detected spikes from SpikeDetector

        Returns:
            List of NeuralFeatures (may be empty if window not yet complete)
        """
        self._spike_buffer.extend(spikes)

        if not spikes:
            return []

        latest_time = spikes[-1].timestamp_sec

        # Initialize window start on first spikes
        if not self._initialized:
            if self._spike_buffer:
                self._next_window_start = self._spike_buffer[0].timestamp_sec
                self._initialized = True
            return []

        completed: List[NeuralFeatures] = []

        # Extract all complete windows
        while self._next_window_start + self.window_size <= latest_time:
            window_end = self._next_window_start + self.window_size
            features = extract_features(
                self._spike_buffer,
                self._next_window_start,
                window_end,
            )
            completed.append(features)
            self._next_window_start += self.window_step

        # Prune old spikes (keep only within 2× window for efficiency)
        cutoff = self._next_window_start - self.window_size
        self._spike_buffer = [s for s in self._spike_buffer if s.timestamp_sec >= cutoff]

        return completed
