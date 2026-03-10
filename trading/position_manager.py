"""
trading/position_manager.py
─────────────────────────────
Orchestrates trade execution: ties together the classifier output,
risk manager, and Hyperliquid client into a clean state machine.

States:
  FLAT   → no open position
  LONG   → holding a long
  SHORT  → holding a short

Transitions:
  FLAT   + LONG signal  → open long
  FLAT   + SHORT signal → open short
  LONG   + SHORT signal → close long, open short (flip)
  SHORT  + LONG signal  → close short, open long (flip)
  ANY    + SL/TP hit    → close position
"""

import time
import logging
from typing import Optional, Literal
from .hyperliquid_client import HyperliquidClient
from .risk import RiskManager
from ..classifier.threshold import ClassifierResult

logger = logging.getLogger(__name__)

PositionState = Literal["FLAT", "LONG", "SHORT"]


class PositionManager:

    def __init__(
        self,
        client: HyperliquidClient,
        risk: RiskManager,
        position_size_usd: float,
    ):
        self._client = client
        self._risk = risk
        self._size_usd = position_size_usd
        self._state: PositionState = "FLAT"
        self._entry_price: Optional[float] = None
        self._last_signal_time: float = 0.0

        # Sync state with exchange on startup
        self._sync_state()

    def _sync_state(self) -> None:
        """Pull current position from exchange to initialize state."""
        if self._client.is_long():
            self._state = "LONG"
            self._entry_price = self._client.get_entry_price()
            logger.info(f"Synced state: LONG @ ${self._entry_price}")
        elif self._client.is_short():
            self._state = "SHORT"
            self._entry_price = self._client.get_entry_price()
            logger.info(f"Synced state: SHORT @ ${self._entry_price}")
        else:
            self._state = "FLAT"
            logger.info("Synced state: FLAT")

    def on_classifier_result(self, result: ClassifierResult) -> Optional[str]:
        """
        Process a classifier result and execute trades as appropriate.

        Returns:
            Description of action taken, or None if no action.
        """
        signal = result.signal

        # ── Check SL/TP on existing position ─────────────────────────────────
        action = self._check_stop_take()
        if action:
            return action

        # ── Signal is HOLD — nothing to do ───────────────────────────────────
        if signal == "HOLD":
            return None

        # ── Same direction as current state — hold ────────────────────────────
        if (signal == "LONG" and self._state == "LONG") or \
           (signal == "SHORT" and self._state == "SHORT"):
            return None

        # ── Risk check before entry ────────────────────────────────────────────
        check = self._risk.check_entry(signal, self._size_usd)
        if not check.allowed:
            logger.debug(f"Trade blocked by risk: {check.reason}")
            return None

        # ── Execute state transition ───────────────────────────────────────────
        if self._state != "FLAT":
            # Close existing position first
            close_action = self._close("Signal flip")
            if close_action:
                logger.info(close_action)

        if signal == "LONG":
            return self._open_long(result.reason)
        elif signal == "SHORT":
            return self._open_short(result.reason)

        return None

    def _open_long(self, reason: str) -> Optional[str]:
        result = self._client.open_long(self._size_usd)
        if result:
            self._state = "LONG"
            self._entry_price = self._client.get_entry_price()
            self._risk.record_entry("LONG", self._entry_price or 0, self._size_usd)
            return f"🟢 LONG opened @ ${self._entry_price:,.2f} — {reason}"
        logger.error("Failed to open LONG position")
        return None

    def _open_short(self, reason: str) -> Optional[str]:
        result = self._client.open_short(self._size_usd)
        if result:
            self._state = "SHORT"
            self._entry_price = self._client.get_entry_price()
            self._risk.record_entry("SHORT", self._entry_price or 0, self._size_usd)
            return f"🔴 SHORT opened @ ${self._entry_price:,.2f} — {reason}"
        logger.error("Failed to open SHORT position")
        return None

    def _close(self, reason: str) -> Optional[str]:
        if self._state == "FLAT":
            return None

        result = self._client.close_position()
        if result:
            exit_price = self._client._get_mid_price() or 0.0
            entry = self._entry_price or exit_price

            if self._state == "LONG":
                pnl = (exit_price - entry) / entry * self._size_usd
            else:
                pnl = (entry - exit_price) / entry * self._size_usd

            self._risk.record_exit(exit_price, pnl, reason)
            prev_state = self._state
            self._state = "FLAT"
            self._entry_price = None

            pnl_sign = "+" if pnl >= 0 else ""
            return (
                f"⬛ {prev_state} closed @ ${exit_price:,.2f} | "
                f"PnL: {pnl_sign}${pnl:.2f} ({reason})"
            )
        return None

    def _check_stop_take(self) -> Optional[str]:
        """Check SL/TP and close if hit."""
        if self._state == "FLAT" or self._entry_price is None:
            return None

        current_price = self._client._get_mid_price()
        if current_price is None:
            return None

        check = self._risk.check_exit(
            entry_price=self._entry_price,
            current_price=current_price,
            is_long=(self._state == "LONG"),
            size_usd=self._size_usd,
        )

        if check.allowed:
            return self._close(check.reason)

        return None

    def force_close(self, reason: str = "Manual force close") -> Optional[str]:
        """Force-close any open position immediately."""
        return self._close(reason)

    @property
    def state(self) -> PositionState:
        return self._state

    @property
    def entry_price(self) -> Optional[float]:
        return self._entry_price
