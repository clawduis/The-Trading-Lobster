"""
monitoring/dashboard.py
────────────────────────
Live terminal dashboard using the Rich library.

Displays:
  - Neural signal stats (spike rate, burst score, current signal)
  - Active position and unrealized PnL
  - Risk engine status
  - Trade history (last 10)
  - System health (hardware connection, uptime)
"""

import time
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich import box

console = Console()


def _signal_color(signal: str) -> str:
    return {"LONG": "green", "SHORT": "red", "HOLD": "yellow"}.get(signal, "white")


def _pnl_color(pnl: float) -> str:
    return "green" if pnl >= 0 else "red"


def _position_color(state: str) -> str:
    return {"LONG": "green", "SHORT": "red", "FLAT": "dim white"}.get(state, "white")


class Dashboard:
    """
    Renders the live trading dashboard.
    Call `update()` with current system state, then `render()` to get the layout.
    """

    def __init__(self):
        self._start_time = time.time()
        self._state = {}

    def update(
        self,
        # Neural
        spike_rate: float = 0.0,
        burst_score: float = 0.0,
        spike_count: int = 0,
        signal: str = "HOLD",
        signal_confidence: float = 0.0,
        signal_reason: str = "",
        # Hardware
        hw_connected: bool = False,
        samples_read: int = 0,
        # Position
        position_state: str = "FLAT",
        entry_price: Optional[float] = None,
        current_price: Optional[float] = None,
        unrealized_pnl: float = 0.0,
        # Risk
        account_value: float = 0.0,
        daily_pnl: float = 0.0,
        total_trades: int = 0,
        wins: int = 0,
        losses: int = 0,
        cooldown_remaining: float = 0.0,
        halted: bool = False,
        halt_reason: str = "",
        # Trade history
        recent_trades: Optional[List[dict]] = None,
    ) -> None:
        self._state = {
            "spike_rate": spike_rate,
            "burst_score": burst_score,
            "spike_count": spike_count,
            "signal": signal,
            "signal_confidence": signal_confidence,
            "signal_reason": signal_reason,
            "hw_connected": hw_connected,
            "samples_read": samples_read,
            "position_state": position_state,
            "entry_price": entry_price,
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
            "account_value": account_value,
            "daily_pnl": daily_pnl,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0.0,
            "cooldown_remaining": cooldown_remaining,
            "halted": halted,
            "halt_reason": halt_reason,
            "recent_trades": recent_trades or [],
            "uptime": time.time() - self._start_time,
        }

    def render(self) -> Layout:
        s = self._state
        uptime_str = _format_duration(s.get("uptime", 0))

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        layout["left"].split_column(
            Layout(name="neural", ratio=2),
            Layout(name="position", ratio=2),
        )
        layout["right"].split_column(
            Layout(name="risk"),
            Layout(name="history", ratio=2),
        )

        # ── Header ────────────────────────────────────────────────────────────
        title = "🦞  THE TRADING LOBSTER  🦞"
        if s.get("halted"):
            title += f"  ⛔ HALTED: {s.get('halt_reason', '')}"
        layout["header"].update(Panel(
            Text(title, justify="center", style="bold cyan"),
            style="cyan",
        ))

        # ── Neural Signal ─────────────────────────────────────────────────────
        sig = s.get("signal", "HOLD")
        sig_color = _signal_color(sig)
        neural_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        neural_table.add_column("Key", style="dim")
        neural_table.add_column("Value")
        neural_table.add_row("Spike Rate",    f"{s.get('spike_rate', 0):.2f} Hz")
        neural_table.add_row("Burst Score",   f"{s.get('burst_score', 0):.3f}")
        neural_table.add_row("Spike Count",   str(s.get("spike_count", 0)))
        neural_table.add_row("Signal",        Text(sig, style=f"bold {sig_color}"))
        neural_table.add_row("Confidence",    f"{s.get('signal_confidence', 0):.2f}")
        neural_table.add_row("Reason",        Text(s.get("signal_reason", "—"), style="dim"))
        hw_status = (
            Text("● CONNECTED", style="green")
            if s.get("hw_connected")
            else Text("● DISCONNECTED", style="red")
        )
        neural_table.add_row("Hardware", hw_status)
        neural_table.add_row("Samples Read",  f"{s.get('samples_read', 0):,}")
        layout["neural"].update(Panel(neural_table, title="[bold]Neural Signal[/bold]", border_style="cyan"))

        # ── Position ──────────────────────────────────────────────────────────
        pos_state = s.get("position_state", "FLAT")
        pos_color = _position_color(pos_state)
        upnl = s.get("unrealized_pnl", 0.0)
        upnl_color = _pnl_color(upnl)
        pos_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        pos_table.add_column("Key", style="dim")
        pos_table.add_column("Value")
        pos_table.add_row("State",          Text(pos_state, style=f"bold {pos_color}"))
        pos_table.add_row("Entry Price",    f"${s.get('entry_price') or 0:,.2f}" if s.get("entry_price") else "—")
        pos_table.add_row("Current Price",  f"${s.get('current_price') or 0:,.2f}" if s.get("current_price") else "—")
        pos_table.add_row("Unrealized PnL", Text(f"${upnl:+.2f}", style=upnl_color))
        layout["position"].update(Panel(pos_table, title="[bold]Position[/bold]", border_style="magenta"))

        # ── Risk ──────────────────────────────────────────────────────────────
        daily_pnl = s.get("daily_pnl", 0.0)
        risk_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        risk_table.add_column("Key", style="dim")
        risk_table.add_column("Value")
        risk_table.add_row("Account Value",    f"${s.get('account_value', 0):,.2f}")
        risk_table.add_row("Daily PnL",        Text(f"${daily_pnl:+.2f}", style=_pnl_color(daily_pnl)))
        risk_table.add_row("Total Trades",     str(s.get("total_trades", 0)))
        risk_table.add_row("Win / Loss",       f"{s.get('wins', 0)} / {s.get('losses', 0)}")
        risk_table.add_row("Win Rate",         f"{s.get('win_rate', 0)*100:.1f}%")
        risk_table.add_row("Cooldown",         f"{s.get('cooldown_remaining', 0):.0f}s")
        layout["risk"].update(Panel(risk_table, title="[bold]Risk & Stats[/bold]", border_style="yellow"))

        # ── Trade History ─────────────────────────────────────────────────────
        history_table = Table(box=box.SIMPLE, padding=(0, 1), show_header=True)
        history_table.add_column("Dir", style="bold", width=6)
        history_table.add_column("Entry", width=10)
        history_table.add_column("Exit", width=10)
        history_table.add_column("PnL", width=10)
        history_table.add_column("Reason", style="dim")

        for trade in reversed(s.get("recent_trades", [])[-10:]):
            direction = trade.get("direction", "?")
            dir_color = "green" if direction == "LONG" else "red"
            pnl = trade.get("pnl")
            pnl_str = f"${pnl:+.2f}" if pnl is not None else "open"
            pnl_style = _pnl_color(pnl) if pnl is not None else "dim"
            history_table.add_row(
                Text(direction, style=dir_color),
                f"${trade.get('entry_price', 0):,.0f}",
                f"${trade.get('exit_price', 0):,.0f}" if trade.get("exit_price") else "—",
                Text(pnl_str, style=pnl_style),
                trade.get("exit_reason", "—"),
            )
        layout["history"].update(Panel(history_table, title="[bold]Trade History[/bold]", border_style="blue"))

        # ── Footer ────────────────────────────────────────────────────────────
        layout["footer"].update(Panel(
            Text(f"Uptime: {uptime_str}   |   Press Ctrl+C to exit gracefully", justify="center", style="dim"),
        ))

        return layout


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
