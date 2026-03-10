"""
scripts/record_baseline.py
───────────────────────────
Records labeled neural data for training the ML classifier.

Usage:
  python scripts/record_baseline.py --label LONG  --duration 120
  python scripts/record_baseline.py --label SHORT --duration 120
  python scripts/record_baseline.py --label HOLD  --duration 60
  python scripts/record_baseline.py --mock         # use synthetic signal

Saves CSV to data/recordings/YYYY-MM-DD_HH-MM-SS_{label}.csv
Columns: timestamp, spike_rate, isi_mean_ms, isi_cv, burst_score,
          amplitude_mean, amplitude_std, label
"""

import argparse
import csv
import os
import sys
import time
import numpy as np
import yaml
from datetime import datetime
from dotenv import load_dotenv

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware.serial_reader import SerialReader
from hardware.mock_reader import MockReader
from processing.filters import NeuralFilter
from processing.spike_detector import SpikeDetector
from processing.features import SlidingWindowFeatureExtractor


def main():
    parser = argparse.ArgumentParser(description="Record labeled neural baseline")
    parser.add_argument("--label",    required=True, choices=["LONG", "SHORT", "HOLD"])
    parser.add_argument("--duration", type=int,   default=120, help="Recording duration in seconds")
    parser.add_argument("--mock",     action="store_true")
    parser.add_argument("--config",   default="config.yaml")
    args = parser.parse_args()

    load_dotenv()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    hw_cfg = cfg["hardware"]
    proc_cfg = cfg["processing"]

    if args.mock:
        reader = MockReader(hw_cfg["sample_rate"], hw_cfg["buffer_size"])
    else:
        reader = SerialReader(
            port=os.getenv("SERIAL_PORT", "/dev/ttyUSB0"),
            baud_rate=int(os.getenv("BAUD_RATE", "115200")),
            sample_rate=hw_cfg["sample_rate"],
            buffer_size=hw_cfg["buffer_size"],
        )

    filt = NeuralFilter(
        sample_rate=hw_cfg["sample_rate"],
        bandpass_low=proc_cfg["bandpass_low"],
        bandpass_high=proc_cfg["bandpass_high"],
        bandpass_order=proc_cfg["bandpass_order"],
        notch_freq=proc_cfg["notch_freq"],
        notch_quality=proc_cfg["notch_quality"],
    )
    detector = SpikeDetector(
        sample_rate=hw_cfg["sample_rate"],
        threshold_mad=proc_cfg["spike_threshold_mad"],
        refractory_ms=proc_cfg["refractory_ms"],
        waveform_window_ms=proc_cfg["spike_window_ms"],
    )
    extractor = SlidingWindowFeatureExtractor(
        window_size_sec=proc_cfg["window_size_sec"],
        window_step_sec=proc_cfg["window_step_sec"],
    )

    os.makedirs("data/recordings", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"data/recordings/{ts}_{args.label}.csv"

    print(f"\n🦞 Recording [{args.label}] for {args.duration}s → {filename}")
    print("   Press Ctrl+C to stop early.\n")

    reader.start()
    start_time = time.time()
    rows_written = 0

    try:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "spike_rate", "isi_mean_ms", "isi_cv",
                "burst_score", "amplitude_mean", "amplitude_std", "label"
            ])

            while time.time() - start_time < args.duration:
                samples = reader.get_samples(hw_cfg["read_chunk"])
                if not samples:
                    time.sleep(0.005)
                    continue

                filtered = filt.process(np.array(samples, dtype=np.float64))
                chunk_time = time.time() - len(samples) / hw_cfg["sample_rate"]
                spikes = detector.detect(filtered, chunk_time)
                windows = extractor.add_spikes(spikes)

                for feat in windows:
                    writer.writerow([
                        feat.window_end,
                        feat.spike_rate,
                        feat.isi_mean_ms,
                        feat.isi_cv,
                        feat.burst_score,
                        feat.amplitude_mean,
                        feat.amplitude_std,
                        args.label,
                    ])
                    rows_written += 1

                elapsed = time.time() - start_time
                remaining = args.duration - elapsed
                print(
                    f"\r  {elapsed:.0f}s / {args.duration}s  |  "
                    f"Windows: {rows_written}  |  "
                    f"Remaining: {remaining:.0f}s  ",
                    end="", flush=True,
                )
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")

    finally:
        reader.stop()

    print(f"\n\n✅ Saved {rows_written} windows to {filename}")


if __name__ == "__main__":
    main()
