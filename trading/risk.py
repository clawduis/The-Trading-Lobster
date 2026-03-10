"""
trading/risk.py
────────────────
Risk management engine.

All trade signals from the classifier pass through this module before
being sent to the exchange. Acts as a stateful gatekeeper.

Checks enforced:
  1. Daily loss limit      — halt if drawdown > max_daily_loss_pct
  2. Max position size     — reject if order would exceed max_position_usd
  3. Cooldown period       — minimum seconds between consecutive trades
  4. Stop loss             — auto-close if unrealized loss > stop_loss_pct
  5. Take profit           — auto-close if unrealized gain > take_profit_pct
  6. Max open positions    — only 1 active position at a time
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List
from ..classifier.threshold import Signal

logger = logging.getLogger(__name__)


@dataclass
class RiskCheck:
    allowed: bool
    reason: str


@dataclass
class TradeRecord:
    timestamp: float
    direction: str          # "LONG" | "SHORT"
    entry_price: float
    size_usd: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None


class RiskManager:
    """
    Stateful risk engine. Must be called before every trade attempt.
    """

    def __init__(
        self,
        max_daily_loss_pct: float,
        max_position_usd: float,
        cooldown_seconds: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        max_open_positions: int,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_usd = max_position_usd
        self.cooldown_seconds = cooldown_seconds
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_open_positions = max_open_positions

        self._session_start = time.time()
        self._starting_balance: Optional[float] = None
        self._last_trade_time: float = 0.0
        self._open_positions: int = 0
        self._halted: bool = False
        self._halt_reason: str = ""
        self._trade_history: List[TradeRecord] = []
        self._daily_pnl: float = 0.0

    def set_starting_balance(self, balance: float) -> None:
        """Call once at startup with the account's USDC balance."""
        self._starting_balance = balance
        logger.info(f"Risk manager: starting balance set to ${balance:.2f}")

    def check_entry(self, signal: Signal, size_usd: float) -> RiskCheck:
        """
        Check whether a new trade entry is permitted.

        Args:
            signal:   "LONG" | "SHORT" | "HOLD"
            size_usd: Proposed trade notional in USD

        Returns:
            RiskCheck(allowed=True/False, reason=...)
        """
        if signal == "HOLD":
            return RiskCheck(allowed=False, reason="Signal is HOLD")

        # ── Hard halt check ───────────────────────────────────────────────────
        if self._halted:
            return RiskCheck(allowed=False, reason=f"Trading halted: {self._halt_reason}")

        # ── Daily loss check ──────────────────────────────────────────────────
        if self._starting_balance and self._starting_balance > 0:
            loss_pct = -self._daily_pnl / self._starting_balance
            if loss_pct >= self.max_daily_loss_pct:
                self._halt(f"Daily loss limit reached: {loss_pct*100:.1f}% >= {self.max_daily_loss_pct*100:.1f}%")
                return RiskCheck(allowed=False, reason=self._halt_reason)

        # ── Cooldown check ────────────────────────────────────────────────────
        elapsed = time.time() - self._last_trade_time
        if elapsed < self.cooldown_seconds:
            remaining = self.cooldown_seconds - elapsed
            return RiskCheck(
                allowed=False,
                reason=f"Cooldown: {remaining:.0f}s remaining"
            )

        # ── Position limit check ───────────────────────────────────────────────
        if self._open_positions >= self.max_open_positions:
            return RiskCheck(
                allowed=False,
                reason=f"Max open positions reached: {self._open_positions}/{self.max_open_positions}"
            )

        # ── Position size check ───────────────────────────────────────────────
        if size_usd > self.max_position_usd:
            return RiskCheck(
                allowed=False,
                reason=f"Position size ${size_usd:.2f} exceeds max ${self.max_position_usd:.2f}"
            )

        return RiskCheck(allowed=True, reason="All risk checks passed")

    def check_exit(
        self,
        entry_price: float,
        current_price: float,
        is_long: bool,
        size_usd: float,
    ) -> RiskCheck:
        """
        Check whether an open position should be force-closed (SL/TP).

        Returns:
            RiskCheck(allowed=True, reason=...) if position should be closed.
            RiskCheck(allowed=False, ...) if position should be held.
        """
        if entry_price <= 0 or current_price <= 0:
            return RiskCheck(allowed=False, reason="Invalid price data")

        pnl_pct = (
            (current_price - entry_price) / entry_price
            if is_long
            else (entry_price - current_price) / entry_price
        )

        if pnl_pct <= -self.stop_loss_pct:
            return RiskCheck(
                allowed=True,
                reason=f"Stop loss: {pnl_pct*100:.2f}% <= -{self.stop_loss_pct*100:.1f}%"
            )

        if pnl_pct >= self.take_profit_pct:
            return RiskCheck(
                allowed=True,
                reason=f"Take profit: {pnl_pct*100:.2f}% >= {self.take_profit_pct*100:.1f}%"
            )

        return RiskCheck(allowed=False, reason=f"Hold: PnL={pnl_pct*100:.2f}%")

    # ─── Bookkeeping ─────────────────────────────────────────────────────────────

    def record_entry(self, direction: str, entry_price: float, size_usd: float) -> None:
        self._last_trade_time = time.time()
        self._open_positions += 1
        self._trade_history.append(TradeRecord(
            timestamp=time.time(),
            direction=direction,
            entry_price=entry_price,
            size_usd=size_usd,
        ))
        logger.info(f"Risk: entry recorded — {direction} ${size_usd:.2f} @ ${entry_price:,.2f}")

    def record_exit(self, exit_price: float, pnl: float, reason: str) -> None:
        self._open_positions = max(0, self._open_positions - 1)
        self._daily_pnl += pnl
        if self._trade_history:
            last = self._trade_history[-1]
            last.exit_price = exit_price
            last.pnl = pnl
            last.exit_reason = reason
        logger.info(f"Risk: exit recorded — PnL=${pnl:.2f} ({reason})")

    def _halt(self, reason: str) -> None:
        self._halted = True
        self._halt_reason = reason
        logger.critical(f"⛔ TRADING HALTED: {reason}")

    def reset_daily(self) -> None:
        """Reset daily counters (call at midnight or session start)."""
        self._daily_pnl = 0.0
        self._halted = False
        self._halt_reason = ""
        logger.info("Risk manager: daily counters reset.")

    # ─── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        wins = [t for t in self._trade_history if t.pnl and t.pnl > 0]
        losses = [t for t in self._trade_history if t.pnl and t.pnl <= 0]
        total_closed = len([t for t in self._trade_history if t.pnl is not None])
        return {
            "daily_pnl": self._daily_pnl,
            "open_positions": self._open_positions,
            "total_trades": len(self._trade_history),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / total_closed if total_closed > 0 else 0.0,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "cooldown_remaining": max(
                0, self.cooldown_seconds - (time.time() - self._last_trade_time)
            ),
        }
