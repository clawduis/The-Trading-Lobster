"""
processing/spike_detector.py
─────────────────────────────
Threshold-based action potential detector.

Method: Median Absolute Deviation (MAD) threshold
  threshold = multiplier × median(|x| / 0.6745)

MAD is robust to non-Gaussian noise and does not require a pre-recorded
baseline — the noise floor is estimated from each window itself.

Reference: Quiroga et al. (2004) — Neural Computation
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Spike:
    """A detected action potential event."""
    sample_index: int           # absolute sample index in the recording
    timestamp_sec: float        # wall-clock time of detection
    amplitude: float            # peak amplitude (ADC units, post-filter)
    waveform: np.ndarray = field(default_factory=lambda: np.array([]))


class SpikeDetector:
    """
    Online spike detector with absolute refractory period enforcement.
    Operates on pre-filtered signal chunks.
    """

    def __init__(
        self,
        sample_rate: int,
        threshold_mad: float,
        refractory_ms: float,
        waveform_window_ms: float,
    ):
        self.sample_rate = sample_rate
        self.threshold_mad = threshold_mad
        self.refractory_samples = int(refractory_ms * sample_rate / 1000)
        self.waveform_half_samples = int(waveform_window_ms * sample_rate / 1000)

        self._total_sample_count = 0
        self._last_spike_sample = -self.refractory_samples  # allow immediate detection
        self._waveform_buffer = np.zeros(self.waveform_half_samples * 2)

    def detect(self, filtered_chunk: np.ndarray, chunk_timestamp: float) -> List[Spike]:
        """
        Detect spikes in a filtered signal chunk.

        Args:
            filtered_chunk: 1-D numpy float array from NeuralFilter.process()
            chunk_timestamp: wall-clock time of the first sample in this chunk

        Returns:
            List of Spike objects detected in this chunk.
        """
        spikes: List[Spike] = []
        n = len(filtered_chunk)

        if n == 0:
            return spikes

        # Estimate noise floor from this chunk using MAD estimator
        noise_floor = np.median(np.abs(filtered_chunk)) / 0.6745
        threshold = self.threshold_mad * noise_floor

        # Find threshold crossings (positive peaks only)
        # Use negative crossings if signal is predominantly negative
        above = filtered_chunk > threshold
        crossings = np.where(np.diff(above.astype(int)) == 1)[0]  # rising edges

        for crossing_idx in crossings:
            abs_idx = self._total_sample_count + crossing_idx

            # Enforce refractory period
            if abs_idx - self._last_spike_sample < self.refractory_samples:
                continue

            # Find peak within next refractory_samples window
            peak_window_end = min(crossing_idx + self.refractory_samples, n)
            peak_offset = np.argmax(filtered_chunk[crossing_idx:peak_window_end])
            peak_idx = crossing_idx + peak_offset
            peak_amplitude = filtered_chunk[peak_idx]

            # Extract waveform (±waveform_window_ms)
            wf_start = max(0, peak_idx - self.waveform_half_samples)
            wf_end = min(n, peak_idx + self.waveform_half_samples)
            waveform = filtered_chunk[wf_start:wf_end].copy()

            abs_peak_idx = self._total_sample_count + peak_idx
            spike_time = chunk_timestamp + (peak_idx / self.sample_rate)

            spikes.append(Spike(
                sample_index=abs_peak_idx,
                timestamp_sec=spike_time,
                amplitude=float(peak_amplitude),
                waveform=waveform,
            ))

            self._last_spike_sample = abs_peak_idx

        self._total_sample_count += n
        return spikes

    def reset(self) -> None:
        """Reset detector state."""
        self._total_sample_count = 0
        self._last_spike_sample = -self.refractory_samples
