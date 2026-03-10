"""
trading/hyperliquid_client.py
──────────────────────────────
Hyperliquid perpetuals trading client.

Uses the official hyperliquid-python-sdk for all order operations.
Handles leverage setup, market open/close, position queries, and account info.

Docs: https://github.com/hyperliquid-dex/hyperliquid-python-sdk
"""

import logging
import time
from typing import Optional

import eth_account
from eth_account.signers.local import LocalAccount

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

logger = logging.getLogger(__name__)


class HyperliquidClient:
    """
    Wrapper around the Hyperliquid SDK for the Trading Lobster.

    Provides:
      - Leverage configuration
      - Market long / short entry
      - Market position close
      - Account value and position queries
    """

    def __init__(
        self,
        private_key: str,
        wallet_address: str,
        use_testnet: bool,
        coin: str,
        leverage: int,
        slippage: float,
    ):
        self.coin = coin
        self.leverage = leverage
        self.slippage = slippage
        self.wallet_address = wallet_address

        api_url = constants.TESTNET_API_URL if use_testnet else constants.MAINNET_API_URL
        network = "TESTNET" if use_testnet else "MAINNET"

        self._account: LocalAccount = eth_account.Account.from_key(private_key)
        self._exchange = Exchange(self._account, api_url)
        self._info = Info(api_url, skip_ws=True)

        logger.info(
            f"HyperliquidClient initialized — {network} | "
            f"coin={coin} | leverage={leverage}x | "
            f"wallet={wallet_address[:8]}..."
        )

    # ─── Leverage ───────────────────────────────────────────────────────────────

    def set_leverage(self) -> bool:
        """Set cross-margin leverage for the configured coin."""
        try:
            result = self._exchange.update_leverage(
                leverage=self.leverage,
                coin=self.coin,
                is_cross=True,
            )
            logger.info(f"Leverage set to {self.leverage}x on {self.coin}: {result}")
            return True
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
            return False

    # ─── Orders ─────────────────────────────────────────────────────────────────

    def open_long(self, size_usd: float) -> Optional[dict]:
        """Open a market long position."""
        return self._market_open(is_buy=True, size_usd=size_usd, direction="LONG")

    def open_short(self, size_usd: float) -> Optional[dict]:
        """Open a market short position."""
        return self._market_open(is_buy=False, size_usd=size_usd, direction="SHORT")

    def _market_open(self, is_buy: bool, size_usd: float, direction: str) -> Optional[dict]:
        """
        Place a market order.

        size_usd is converted to coin size using current mid price.
        """
        try:
            mid_price = self._get_mid_price()
            if mid_price is None or mid_price <= 0:
                logger.error("Cannot open position: failed to get current price")
                return None

            coin_size = round(size_usd / mid_price, 6)
            if coin_size <= 0:
                logger.error(f"Computed coin size is zero or negative: {coin_size}")
                return None

            logger.info(
                f"Opening {direction}: {coin_size:.6f} {self.coin} "
                f"(~${size_usd:.2f} @ ${mid_price:,.2f})"
            )

            result = self._exchange.market_open(
                coin=self.coin,
                is_buy=is_buy,
                sz=coin_size,
                slippage=self.slippage,
            )

            logger.info(f"Order result: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to open {direction} position: {e}")
            return None

    def close_position(self) -> Optional[dict]:
        """Close the current open position (if any)."""
        try:
            position = self.get_position()
            if position is None:
                logger.info("No open position to close.")
                return None

            szi = float(position.get("szi", 0))
            if szi == 0:
                logger.info("Position size is zero, nothing to close.")
                return None

            logger.info(f"Closing position: szi={szi} {self.coin}")

            result = self._exchange.market_close(
                coin=self.coin,
                slippage=self.slippage,
            )

            logger.info(f"Close result: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return None

    # ─── Account / Market Info ───────────────────────────────────────────────────

    def get_position(self) -> Optional[dict]:
        """
        Get the current open position for the configured coin.

        Returns:
            Position dict with keys: coin, szi, entryPx, unrealizedPnl, liquidationPx
            or None if no position is open.
        """
        try:
            state = self._info.user_state(self.wallet_address)
            for pos in state.get("assetPositions", []):
                p = pos.get("position", {})
                if p.get("coin") == self.coin:
                    szi = float(p.get("szi", 0))
                    if szi != 0:
                        return p
            return None
        except Exception as e:
            logger.error(f"Failed to get position: {e}")
            return None

    def get_account_value(self) -> Optional[float]:
        """Return total account value in USDC."""
        try:
            state = self._info.user_state(self.wallet_address)
            summary = state.get("marginSummary", {})
            return float(summary.get("accountValue", 0))
        except Exception as e:
            logger.error(f"Failed to get account value: {e}")
            return None

    def get_unrealized_pnl(self) -> float:
        """Return unrealized PnL of current position (0 if flat)."""
        try:
            position = self.get_position()
            if position is None:
                return 0.0
            return float(position.get("unrealizedPnl", 0))
        except Exception as e:
            logger.error(f"Failed to get unrealized PnL: {e}")
            return 0.0

    def get_entry_price(self) -> Optional[float]:
        """Return entry price of current position."""
        position = self.get_position()
        if position is None:
            return None
        return float(position.get("entryPx", 0)) or None

    def _get_mid_price(self) -> Optional[float]:
        """Fetch current mid price for the configured coin."""
        try:
            mids = self._info.all_mids()
            price_str = mids.get(self.coin)
            if price_str is None:
                logger.error(f"No mid price found for {self.coin}")
                return None
            return float(price_str)
        except Exception as e:
            logger.error(f"Failed to fetch mid price: {e}")
            return None

    def is_long(self) -> bool:
        """Return True if currently holding a long position."""
        pos = self.get_position()
        if pos is None:
            return False
        return float(pos.get("szi", 0)) > 0

    def is_short(self) -> bool:
        """Return True if currently holding a short position."""
        pos = self.get_position()
        if pos is None:
            return False
        return float(pos.get("szi", 0)) < 0

    def is_flat(self) -> bool:
        return self.get_position() is None
