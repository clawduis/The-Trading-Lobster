"""
main.py — The Trading Lobster
──────────────────────────────
Entry point. Wires together the full pipeline:

  Hardware → Filter → Spike Detection → Feature Extraction
  → Classification → Risk Check → Hyperliquid Trade Execution
  → Dashboard

Usage:
  python main.py                    # run with real hardware + live trading
  python main.py --mock             # use synthetic signal generator
  python main.py --dry-run          # real signal, no trades executed
  python main.py --mock --dry-run   # fully offline dev mode
"""

import argparse
import os
import sys
import time
import signal
import logging
import numpy as np
import yaml
from dotenv import load_dotenv
from rich.live import Live

from hardware.serial_reader import SerialReader
from hardware.mock_reader import MockReader
from processing.filters import NeuralFilter
from processing.spike_detector import SpikeDetector
from processing.features import SlidingWindowFeatureExtractor
from classifier.threshold import ThresholdClassifier
from classifier.ml_classifier import MLClassifier
from trading.hyperliquid_client import HyperliquidClient
from trading.risk import RiskManager
from trading.position_manager import PositionManager
from monitoring.logger import setup_logging
from monitoring.dashboard import Dashboard

logger = logging.getLogger(__name__)

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info(f"Received signal {signum} — shutting down...")
    _shutdown = True


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_classifier(cfg: dict):
    mode = cfg["classifier"]["mode"]
    if mode == "ml":
        return MLClassifier(
            model_path=cfg["classifier"]["ml"]["model_path"],
            min_confidence=cfg["classifier"]["ml"]["min_confidence"],
        )
    else:
        t = cfg["classifier"]["threshold"]
        return ThresholdClassifier(
            long_spike_rate=t["long_spike_rate"],
            short_spike_rate=t["short_spike_rate"],
            long_burst_score=t["long_burst_score"],
            short_burst_score=t["short_burst_score"],
            min_confidence=t["min_confidence"],
        )


def main():
    global _shutdown

    # ── Args ────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="The Trading Lobster")
    parser.add_argument("--mock",    action="store_true", help="Use synthetic signal")
    parser.add_argument("--dry-run", action="store_true", help="Classify but do not trade")
    parser.add_argument("--config",  default="config.yaml", help="Config file path")
    args = parser.parse_args()

    # ── Config & env ─────────────────────────────────────────────────────────────
    load_dotenv()
    cfg = load_config(args.config)

    setup_logging(
        log_level=cfg["monitoring"]["log_level"],
        log_file=cfg["monitoring"]["log_file"],
    )

    logger.info("=" * 60)
    logger.info("  🦞  THE TRADING LOBSTER — STARTING UP")
    logger.info("=" * 60)
    logger.info(f"  Mock mode:  {args.mock}")
    logger.info(f"  Dry run:    {args.dry_run}")
    logger.info(f"  Config:     {args.config}")

    # ── Hardware ─────────────────────────────────────────────────────────────────
    hw_cfg = cfg["hardware"]
    use_mock = args.mock or hw_cfg.get("use_mock", False)

    if use_mock:
        reader = MockReader(
            sample_rate=hw_cfg["sample_rate"],
            buffer_size=hw_cfg["buffer_size"],
        )
        logger.info("Hardware: MockReader (synthetic signal)")
    else:
        serial_port = os.getenv("SERIAL_PORT", "/dev/ttyUSB0")
        baud_rate = int(os.getenv("BAUD_RATE", str(hw_cfg.get("baud_rate", 115200))))
        reader = SerialReader(
            port=serial_port,
            baud_rate=baud_rate,
            sample_rate=hw_cfg["sample_rate"],
            buffer_size=hw_cfg["buffer_size"],
        )
        logger.info(f"Hardware: SerialReader on {serial_port}")

    # ── Signal Processing ─────────────────────────────────────────────────────────
    proc_cfg = cfg["processing"]
    neural_filter = NeuralFilter(
        sample_rate=hw_cfg["sample_rate"],
        bandpass_low=proc_cfg["bandpass_low"],
        bandpass_high=proc_cfg["bandpass_high"],
        bandpass_order=proc_cfg["bandpass_order"],
        notch_freq=proc_cfg["notch_freq"],
        notch_quality=proc_cfg["notch_quality"],
    )

    spike_detector = SpikeDetector(
        sample_rate=hw_cfg["sample_rate"],
        threshold_mad=proc_cfg["spike_threshold_mad"],
        refractory_ms=proc_cfg["refractory_ms"],
        waveform_window_ms=proc_cfg["spike_window_ms"],
    )

    feature_extractor = SlidingWindowFeatureExtractor(
        window_size_sec=proc_cfg["window_size_sec"],
        window_step_sec=proc_cfg["window_step_sec"],
    )

    # ── Classifier ────────────────────────────────────────────────────────────────
    classifier = build_classifier(cfg)
    logger.info(f"Classifier: {cfg['classifier']['mode']} mode")

    # ── Trading ───────────────────────────────────────────────────────────────────
    risk_cfg = cfg["risk"]
    trade_cfg = cfg["trading"]

    risk_manager = RiskManager(
        max_daily_loss_pct=risk_cfg["max_daily_loss_pct"],
        max_position_usd=risk_cfg["max_position_usd"],
        cooldown_seconds=risk_cfg["cooldown_seconds"],
        stop_loss_pct=risk_cfg["stop_loss_pct"],
        take_profit_pct=risk_cfg["take_profit_pct"],
        max_open_positions=risk_cfg["max_open_positions"],
    )

    if not args.dry_run:
        private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
        wallet_address = os.getenv("HYPERLIQUID_WALLET_ADDRESS")
        use_testnet = os.getenv("USE_TESTNET", "true").lower() == "true"

        if not private_key or not wallet_address:
            logger.error("HYPERLIQUID_PRIVATE_KEY and HYPERLIQUID_WALLET_ADDRESS must be set in .env")
            sys.exit(1)

        hl_client = HyperliquidClient(
            private_key=private_key,
            wallet_address=wallet_address,
            use_testnet=use_testnet,
            coin=trade_cfg["coin"],
            leverage=trade_cfg["leverage"],
            slippage=trade_cfg["slippage"],
        )
        hl_client.set_leverage()

        account_value = hl_client.get_account_value() or 0.0
        risk_manager.set_starting_balance(account_value)
        logger.info(f"Account value: ${account_value:.2f}")

        position_manager = PositionManager(
            client=hl_client,
            risk=risk_manager,
            position_size_usd=trade_cfg["position_size_usd"],
        )
    else:
        hl_client = None
        position_manager = None
        logger.info("Dry-run mode: trading disabled")

    # ── Dashboard ─────────────────────────────────────────────────────────────────
    dashboard = Dashboard()
    refresh_interval = 1.0 / cfg["monitoring"]["dashboard_refresh_hz"]

    # ── Signal handlers for graceful shutdown ─────────────────────────────────────
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ── Main loop ─────────────────────────────────────────────────────────────────
    reader.start()
    last_dashboard_update = 0.0
    last_result = None
    total_spikes = 0
    all_spikes = []

    logger.info("🦞 Main loop started — waiting for neural signal...")

    with Live(dashboard.render(), refresh_per_second=cfg["monitoring"]["dashboard_refresh_hz"]) as live:
        while not _shutdown:
            loop_start = time.time()

            # 1. Read samples from hardware
            samples = reader.get_samples(hw_cfg["read_chunk"])
            if not samples:
                time.sleep(0.01)
                continue

            # 2. Filter
            raw = np.array(samples, dtype=np.float64)
            filtered = neural_filter.process(raw)

            # 3. Detect spikes
            chunk_time = time.time() - (len(samples) / hw_cfg["sample_rate"])
            new_spikes = spike_detector.detect(filtered, chunk_time)
            all_spikes.extend(new_spikes)
            total_spikes += len(new_spikes)

            # Trim spike buffer to last 2 windows
            cutoff = time.time() - proc_cfg["window_size_sec"] * 2
            all_spikes = [s for s in all_spikes if s.timestamp_sec >= cutoff]

            # 4. Extract features
            feature_windows = feature_extractor.add_spikes(new_spikes)

            for features in feature_windows:
                # 5. Classify
                result = classifier.classify(features)
                last_result = result

                if result.signal != "HOLD":
                    logger.info(
                        f"Signal: {result.signal} | "
                        f"Confidence: {result.confidence:.2f} | "
                        f"Rate: {result.spike_rate:.1f}Hz | "
                        f"Burst: {result.burst_score:.2f} | "
                        f"{result.reason}"
                    )

                # 6. Execute trade
                if position_manager:
                    action = position_manager.on_classifier_result(result)
                    if action:
                        logger.info(f"Trade action: {action}")

            # 7. Update dashboard
            now = time.time()
            if now - last_dashboard_update >= refresh_interval:
                # Gather state for dashboard
                hw_stats = reader.stats()
                risk_stats = risk_manager.stats()

                current_price = hl_client._get_mid_price() if hl_client else None
                unrealized_pnl = hl_client.get_unrealized_pnl() if hl_client else 0.0
                account_value = hl_client.get_account_value() if hl_client else 0.0
                pos_state = position_manager.state if position_manager else "FLAT"
                entry_price = position_manager.entry_price if position_manager else None

                # Build recent trades for display
                recent_trades = []
                for t in risk_manager._trade_history[-10:]:
                    recent_trades.append({
                        "direction": t.direction,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "pnl": t.pnl,
                        "exit_reason": t.exit_reason,
                    })

                dashboard.update(
                    spike_rate=last_result.spike_rate if last_result else 0.0,
                    burst_score=last_result.burst_score if last_result else 0.0,
                    spike_count=total_spikes,
                    signal=last_result.signal if last_result else "HOLD",
                    signal_confidence=last_result.confidence if last_result else 0.0,
                    signal_reason=last_result.reason if last_result else "—",
                    hw_connected=hw_stats.get("connected", False),
                    samples_read=hw_stats.get("samples_read", 0),
                    position_state=pos_state,
                    entry_price=entry_price,
                    current_price=current_price,
                    unrealized_pnl=unrealized_pnl,
                    account_value=account_value or 0.0,
                    daily_pnl=risk_stats["daily_pnl"],
                    total_trades=risk_stats["total_trades"],
                    wins=risk_stats["wins"],
                    losses=risk_stats["losses"],
                    cooldown_remaining=risk_stats["cooldown_remaining"],
                    halted=risk_stats["halted"],
                    halt_reason=risk_stats["halt_reason"],
                    recent_trades=recent_trades,
                )
                live.update(dashboard.render())
                last_dashboard_update = now

            # Yield CPU between iterations
            elapsed = time.time() - loop_start
            sleep_time = max(0, 0.005 - elapsed)  # target ~200Hz loop
            time.sleep(sleep_time)

    # ── Shutdown ──────────────────────────────────────────────────────────────────
    logger.info("Shutdown initiated...")

    if position_manager:
        action = position_manager.force_close("Graceful shutdown")
        if action:
            logger.info(f"Shutdown close: {action}")

    reader.stop()
    logger.info("🦞 The Trading Lobster has gone to sleep. Goodbye.")


if __name__ == "__main__":
    main()
