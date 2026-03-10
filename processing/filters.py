"""
processing/filters.py
─────────────────────
Signal conditioning for raw neural recordings.

Pipeline:
  1. Bandpass filter (300–3000 Hz) — isolates action potential band
  2. Notch filter (60 Hz or 50 Hz) — removes powerline interference
  3. Normalization to z-score for threshold-independent spike detection
"""

import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class NeuralFilter:
    """
    Stateful filter chain for continuous neural signal processing.
    Uses second-order sections (SOS) for numerical stability.
    """

    def __init__(
        self,
        sample_rate: int,
        bandpass_low: float,
        bandpass_high: float,
        bandpass_order: int,
        notch_freq: float,
        notch_quality: float,
    ):
        self.sample_rate = sample_rate
        nyq = sample_rate / 2.0

        # ── Bandpass (Butterworth) ──────────────────────────────────────────────
        low = bandpass_low / nyq
        high = bandpass_high / nyq
        if not (0 < low < 1 and 0 < high < 1 and low < high):
            raise ValueError(
                f"Invalid bandpass frequencies: [{bandpass_low}, {bandpass_high}] Hz "
                f"for sample_rate={sample_rate} Hz"
            )
        self._bp_sos = signal.butter(
            bandpass_order, [low, high], btype="bandpass", output="sos"
        )

        # ── Notch (IIR) ────────────────────────────────────────────────────────
        b_notch, a_notch = signal.iirnotch(notch_freq / nyq, notch_quality)
        self._notch_sos = signal.tf2sos(b_notch, a_notch)

        # ── Filter state (for continuous processing across chunks) ──────────────
        self._bp_zi = signal.sosfilt_zi(self._bp_sos)
        self._notch_zi = signal.sosfilt_zi(self._notch_sos)
        self._initialized = False

        logger.debug(
            f"NeuralFilter initialized: BP=[{bandpass_low},{bandpass_high}] Hz, "
            f"Notch={notch_freq} Hz, fs={sample_rate} Hz"
        )

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Filter a chunk of raw samples.

        Args:
            samples: 1-D numpy array of raw ADC values (int or float)

        Returns:
            Filtered signal as float64 array, same length as input.
        """
        x = samples.astype(np.float64)

        if not self._initialized:
            # Scale initial conditions to signal mean to prevent startup transient
            mean = np.mean(x[:min(len(x), 64)])
            n_bp = self._bp_sos.shape[0]
            n_notch = self._notch_sos.shape[0]
            self._bp_zi = self._bp_zi * mean
            self._notch_zi = self._notch_zi * mean
            self._initialized = True

        # Bandpass
        x, self._bp_zi = signal.sosfilt(self._bp_sos, x, zi=self._bp_zi)

        # Notch
        x, self._notch_zi = signal.sosfilt(self._notch_sos, x, zi=self._notch_zi)

        return x

    def reset(self) -> None:
        """Reset filter state (call when signal is interrupted/reconnected)."""
        self._bp_zi = signal.sosfilt_zi(self._bp_sos)
        self._notch_zi = signal.sosfilt_zi(self._notch_sos)
        self._initialized = False
