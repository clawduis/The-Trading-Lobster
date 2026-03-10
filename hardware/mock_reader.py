"""
hardware/mock_reader.py
───────────────────────
Simulates a lobster neural recording for development and testing.

Generates realistic synthetic spike trains using a Poisson process,
alternating between high-activity (LONG) and low-activity (SHORT) states
to exercise the full classification and trading pipeline without hardware.
"""

import threading
import time
import math
import random
import collections
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Synthetic neural signal parameters
BASELINE_NOISE_STD = 200     # ADC units — electrode noise floor
SPIKE_AMPLITUDE = 2000       # ADC units — action potential amplitude
SPIKE_WIDTH_SAMPLES = 30     # samples — AP duration at 30kHz
STATE_DURATION_SEC = 60      # seconds per behavioral state before switching


def _gaussian_spike(n_samples: int, amplitude: int) -> list:
    """Generate a single action potential waveform (biphasic Gaussian)."""
    t = [i / n_samples for i in range(n_samples)]
    waveform = []
    for x in t:
        # Biphasic shape: positive peak then negative trough
        phase1 = amplitude * math.exp(-((x - 0.3) ** 2) / 0.01)
        phase2 = -amplitude * 0.4 * math.exp(-((x - 0.6) ** 2) / 0.015)
        waveform.append(int(phase1 + phase2))
    return waveform


class MockReader:
    """
    Drop-in replacement for SerialReader that generates synthetic neural data.
    Alternates between HIGH (→ LONG) and LOW (→ SHORT) firing rate states.
    """

    def __init__(self, sample_rate: int, buffer_size: int):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        self._buffer: collections.deque = collections.deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.samples_read = 0
        self.read_errors = 0
        self.connected = False

        # Synthetic state machine
        self._state = "HIGH"               # HIGH or LOW firing
        self._state_start = time.time()
        self._spike_template = _gaussian_spike(SPIKE_WIDTH_SAMPLES, SPIKE_AMPLITUDE)

    @property
    def _current_spike_rate(self) -> float:
        """Return spike rate (Hz) for current state."""
        elapsed = time.time() - self._state_start
        if elapsed > STATE_DURATION_SEC:
            self._state = "LOW" if self._state == "HIGH" else "HIGH"
            self._state_start = time.time()
            logger.debug(f"[MockReader] State switched to {self._state}")
        return 18.0 if self._state == "HIGH" else 3.5

    def start(self) -> None:
        self.connected = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._generate_loop, daemon=True, name="MockReader"
        )
        self._thread.start()
        logger.info("[MockReader] Started synthetic neural signal generator.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self.connected = False
        logger.info("[MockReader] Stopped.")

    def _generate_loop(self) -> None:
        """Generate synthetic samples in real time."""
        chunk_size = 512          # samples per generation tick
        tick_interval = chunk_size / self.sample_rate   # seconds per tick
        spike_cooldown = 0        # samples until next spike is allowed

        while not self._stop_event.is_set():
            rate = self._current_spike_rate
            samples = []

            for _ in range(chunk_size):
                # Background noise
                noise = int(random.gauss(0, BASELINE_NOISE_STD))

                # Probabilistic spike injection
                spike_prob_per_sample = rate / self.sample_rate
                if spike_cooldown <= 0 and random.random() < spike_prob_per_sample:
                    # Insert spike template starting here
                    samples.append(noise + self._spike_template[0])
                    spike_cooldown = SPIKE_WIDTH_SAMPLES + int(
                        self.sample_rate * 0.001
                    )  # 1ms refractory
                else:
                    samples.append(noise)

                spike_cooldown = max(0, spike_cooldown - 1)

            with self._lock:
                self._buffer.extend(samples)
            self.samples_read += chunk_size
            time.sleep(tick_interval)

    def get_samples(self, n: int) -> list:
        with self._lock:
            buf = list(self._buffer)
        return buf[-n:] if len(buf) >= n else buf

    def drain(self) -> list:
        with self._lock:
            samples = list(self._buffer)
            self._buffer.clear()
        return samples

    def buffer_len(self) -> int:
        with self._lock:
            return len(self._buffer)

    def stats(self) -> dict:
        return {
            "samples_read": self.samples_read,
            "read_errors": self.read_errors,
            "buffer_len": self.buffer_len(),
            "connected": self.connected,
            "mock_state": self._state,
            "mock_rate_hz": self._current_spike_rate,
        }
