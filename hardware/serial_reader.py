"""
hardware/serial_reader.py
─────────────────────────
Reads raw ADC samples from a serial-connected bio-amplifier or Arduino.

Expected serial protocol (from Arduino firmware):
  - Each line: a single signed integer ADC value followed by '\n'
  - Example output: "1024\n", "-512\n", "2048\n"
  - Sampling rate must match config.hardware.sample_rate

Compatible hardware:
  - Custom Arduino + INA333 instrumentation amp
  - OpenBCI Cyton (8-channel, configure channel 1)
  - Intan RHD2132 via USB adapter
"""

import threading
import time
import collections
import serial
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SerialReader:
    """
    Background thread that continuously reads ADC samples from a serial port
    and fills a thread-safe ring buffer.
    """

    def __init__(self, port: str, baud_rate: int, sample_rate: int, buffer_size: int):
        self.port = port
        self.baud_rate = baud_rate
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        self._buffer: collections.deque = collections.deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._serial: Optional[serial.Serial] = None

        # Statistics
        self.samples_read = 0
        self.read_errors = 0
        self.connected = False

    def start(self) -> None:
        """Open serial port and start the background reader thread."""
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=1.0,
            )
            self.connected = True
            logger.info(f"Serial port opened: {self.port} @ {self.baud_rate} baud")
        except serial.SerialException as e:
            logger.error(f"Failed to open serial port {self.port}: {e}")
            raise

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name="SerialReader")
        self._thread.start()

    def stop(self) -> None:
        """Signal the reader thread to stop and close the serial port."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._serial and self._serial.is_open:
            self._serial.close()
        self.connected = False
        logger.info("Serial reader stopped.")

    def _read_loop(self) -> None:
        """Main read loop — runs in background thread."""
        while not self._stop_event.is_set():
            try:
                if self._serial.in_waiting > 0:
                    raw = self._serial.readline().decode("utf-8", errors="ignore").strip()
                    if raw:
                        try:
                            value = int(raw)
                            with self._lock:
                                self._buffer.append(value)
                            self.samples_read += 1
                        except ValueError:
                            # Non-numeric line (could be init message from Arduino)
                            logger.debug(f"Non-numeric serial data ignored: {raw!r}")
                else:
                    time.sleep(0.0001)  # 100µs yield to prevent CPU spin
            except serial.SerialException as e:
                self.read_errors += 1
                logger.warning(f"Serial read error: {e}. Attempting reconnect...")
                self._reconnect()
            except Exception as e:
                self.read_errors += 1
                logger.error(f"Unexpected error in serial reader: {e}")

    def _reconnect(self) -> None:
        """Attempt to reconnect the serial port with backoff."""
        self.connected = False
        for attempt in range(1, 6):
            time.sleep(2 ** attempt)  # exponential backoff: 2, 4, 8, 16, 32 sec
            try:
                if self._serial:
                    self._serial.close()
                self._serial = serial.Serial(
                    port=self.port,
                    baudrate=self.baud_rate,
                    timeout=1.0,
                )
                self.connected = True
                logger.info(f"Reconnected to {self.port} on attempt {attempt}")
                return
            except serial.SerialException as e:
                logger.warning(f"Reconnect attempt {attempt} failed: {e}")
        logger.error(f"Could not reconnect to {self.port} after 5 attempts. Stopping reader.")
        self._stop_event.set()

    def get_samples(self, n: int) -> list:
        """
        Retrieve the last `n` samples from the buffer.
        Returns fewer than `n` samples if the buffer has not filled yet.
        """
        with self._lock:
            buf = list(self._buffer)
        return buf[-n:] if len(buf) >= n else buf

    def drain(self) -> list:
        """Drain and return all samples currently in the buffer."""
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
        }
