"""Production-grade Binance USDⓈ-M Futures trading bot.

Strategy: Altcoin Daily Volume Rotation + Donchian 120 breakout
          + 4.0x Dynamic ATR Trailing Stop + Reversal Exit.

Flow
----
1. Once every 24h at 00:00 UTC, scan all USDⓈ-M perpetuals and pick the
   single highest 24h-quote-volume altcoin (USDT margin, BTC excluded).
2. Every ``POLL_SECONDS``, fetch the last 150 ``TIMEFRAME`` candles for that
   symbol and compute Donchian-120 upper/lower bands (current candle
   excluded, no look-ahead) plus a 14-period ATR.
3. Long entry  : close breaks above upper band.
   Short entry : close breaks below lower band.
4. Position sizing (ATR-volatility adjusted):
       Target_Notional = (Equity * RISK_FACTOR) * Price / ATR
       capped at Equity * LEVERAGE_CAP.
   Converted to base-asset quantity via the market's amount precision and
   clamped to the market's min/max amount limits.
5. Execution:
   * Entries are LIMIT orders with ``postOnly=True`` (maker fee 0.02%).
   * On fill, a hard STOP_MARKET is placed 4.0x ATR away from entry.
   * The stop is actively trailed as price moves in our favor.
   * If the opposite Donchian band is touched, exit immediately via MARKET
     order (taker fee 0.05%) and cancel the trailing stop.

Credentials are loaded from environment variables or a local ``.env`` file.
Set ``USE_TESTNET = True`` in the ``.env`` (``USE_TESTNET`` key) or below to
paper-trade on Binance Testnet.

Run:  python binance_volume_rotator_bot.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import ccxt
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich import box
from datetime import datetime

load_dotenv()  # must run BEFORE the module-level config reads below

# --------------------------------------------------------------------------
# Strategy / risk configuration (overridable via .env)
# --------------------------------------------------------------------------
USE_TESTNET = os.getenv("USE_TESTNET", "True").strip().lower() in {
    "1", "true", "yes", "on",
}  # safe default: paper-trading on Testnet
RISK_FACTOR = float(os.getenv("RISK_FACTOR", "0.05"))          # 5.0% of equity at risk
LEVERAGE_CAP = float(os.getenv("LEVERAGE_CAP", "10.0"))        # hard ceiling
DONCHIAN_PERIOD = int(os.getenv("DONCHIAN_PERIOD", "120"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_STOP_MULTIPLIER = float(os.getenv("ATR_STOP_MULTIPLIER", "4.0"))
TIMEFRAME = os.getenv("TIMEFRAME", "4h")
NUM_ASSETS_TO_TRADE = int(os.getenv("NUM_ASSETS_TO_TRADE", "1"))  # only #1 volume asset

# Operational tuning
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))             # candle polling loop
CANDLE_HISTORY_LIMIT = 150                                      # >= DONCHIAN_PERIOD + ATR warmup
DAILY_SCAN_WINDOW_SECONDS = 60                                  # re-scan at 00:00 UTC +- this
LOG_FILE = "bot_run.log"
MAKER_FEE = 0.0002                                              # 0.02%
TAKER_FEE = 0.0005                                              # 0.05%

# Trading pair of the margin collateral / notional denominator.
COLLATERAL = "USDT"

logger = logging.getLogger("binance_volume_rotator")


def _configure_logging() -> None:
    """Console + file logging with a consistent format."""
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))
    except OSError as exc:  # pragma: no cover - log file unwritable, keep console
        logging.basicConfig(level=logging.INFO, format=fmt)
        logger.warning("Cannot open log file %s: %s", LOG_FILE, exc)
        return
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


@dataclass
class SignalSnapshot:
    """Computed indicators for the active symbol at one point in time."""

    symbol: str
    price: float
    atr: float
    donchian_upper: float
    donchian_lower: float
    long_signal: bool
    short_signal: bool
    reversal_long: bool   # opposite lower band touched -> exit a long
    reversal_short: bool  # opposite upper band touched -> exit a short


class BinanceVolumeRotatorBot:
    """Main bot: scans, signals, sizes, executes, and trails stops."""

    def __init__(self) -> None:
        _configure_logging()
        self.console = Console()
        self.exchange = self._build_exchange()
        self.active_symbol: Optional[str] = None
        self.position_side: Optional[str] = None      # "long" | "short" | None
        self.position_amount: float = 0.0
        self.entry_price: float = 0.0
        self.trailing_stop_price: float = 0.0
        self.trailing_stop_order_id: Optional[str] = None
        self.last_daily_scan_ts: Optional[int] = None
        self.markets_cache: Dict[str, dict] = {}
        self.last_candles: List[Tuple] = []  # store for chart
        self.last_snap: Optional[SignalSnapshot] = None
        self.last_balance: float = 0.0

    # ------------------------------------------------------------------ cfg
    def _build_exchange(self) -> ccxt.Exchange:
        """Construct the Binance USDⓈ-M futures client."""
        load_dotenv()  # no-op if .env absent; env vars take precedence
        api_key = os.getenv("API_KEY") or os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("API_SECRET") or os.getenv("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            raise RuntimeError(
                "Missing API credentials. Set API_KEY / API_SECRET (or "
                "BINANCE_API_KEY / BINANCE_API_SECRET) in the environment "
                "or a .env file."
            )

        params: dict = {
            "apiKey": api_key,
            "secret": api_secret,
            "options": {
                "fetchCurrencies": False,  # Avoid sapi endpoints not available on testnet
            }
        }
        # Use binanceusdm for USDⓈ-M futures.
        exchange = ccxt.binanceusdm(params)
        # Explicitly set sandbox flag and override URLs if testnet.
        if USE_TESTNET:
            exchange.sandbox = True
            # Provide complete API URL mapping for testnet/demo trading
            exchange.urls = {
                'api': {
                    'public': 'https://testnet.binance.vision/api/v3',
                    'private': 'https://testnet.binance.vision/api/v3',
                    'v1': 'https://testnet.binance.vision/api/v1',
                    'fapiPublic': 'https://testnet.binancefuture.com/fapi/v1',
                    'fapiPublicV2': 'https://testnet.binancefuture.com/fapi/v2',
                    'fapiPublicV3': 'https://testnet.binancefuture.com/fapi/v3',
                    'fapiPrivate': 'https://testnet.binancefuture.com/fapi/v1',
                    'fapiPrivateV2': 'https://testnet.binancefuture.com/fapi/v2',
                    'fapiPrivateV3': 'https://testnet.binancefuture.com/fapi/v3',
                }
            }
            # Override set_sandbox_mode to no-op since we handle it manually
            exchange.set_sandbox_mode = lambda *args, **kwargs: None
            logger.info("Binance exchange using demo trading URLs: %s", exchange.urls)
        else:
            logger.info("Binance exchange production URLs: %s", exchange.urls.get("api"))
        # Load markets eagerly so amount/price precision is available.
        exchange.load_markets()
        return exchange

    # ------------------------------------------------------------ utilities
    def _fetch_balance_usdt(self) -> float:
        """Available equity in USDT (wallet balance on the futures account)."""
        balance = self.exchange.fetch_balance()
        # CCXT normalizes per-currency; fall back to 'free' USDT if no total.
        usdt = balance.get("USDT") or {}
        equity = float(usdt.get("total") or usdt.get("free") or 0.0)
        if equity <= 0:
            raise RuntimeError("Zero/negative USDT equity; cannot size a position.")
        logger.info("Account equity: %.2f USDT", equity)
        return equity

    def _market(self, symbol: str) -> dict:
        """Cached market metadata for `symbol`."""
        if symbol not in self.markets_cache:
            self.markets_cache[symbol] = self.exchange.market(symbol)
        return self.markets_cache[symbol]

    def _amount_to_precision(self, symbol: str, amount: float) -> float:
        """Round base-asset quantity to the market's amount precision."""
        market = self._market(symbol)
        precision = market.get("precision", {}).get("amount")
        if precision is None:
            return amount
        return float(self.exchange.amount_to_precision(symbol, amount))

    def _clamp_amount(self, symbol: str, amount: float) -> float:
        """Clamp quantity to the market's min/max amount limits."""
        market = self._market(symbol)
        limits = market.get("limits", {}).get("amount", {})
        min_amount = float(limits.get("min") or 0.0)
        max_amount = float(limits.get("max") or float("inf"))
        if min_amount and amount < min_amount:
            logger.warning("Qty %.6f < min_amount %.6f; raising to min.", amount, min_amount)
            amount = min_amount
        if amount > max_amount:
            logger.warning("Qty %.6f > max_amount %.6f; capping.", amount, max_amount)
            amount = max_amount
        return self._amount_to_precision(symbol, amount)

    # ------------------------------------------------------- daily rotation
    def get_top_volume_altcoin(self) -> str:
        """Return the single highest 24h-quoteVolume USDT-margined altcoin.

        Uses 24h tickers, filters to ``/USDT:USDT`` perpetuals, excludes
        ``BTC/USDT:USDT``, then sorts by ``quoteVolume`` descending. The
        returned volume is for the last 24h as reported by the exchange.
        """
        tickers = self.exchange.fetch_tickers()
        candidates: Dict[str, float] = {}
        for symbol, ticker in tickers.items():
            if not symbol.endswith("/USDT:USDT"):
                continue
            if symbol == "BTC/USDT:USDT":
                continue
            quote_volume = ticker.get("quoteVolume")
            if quote_volume is None:
                continue
            try:
                candidates[symbol] = float(quote_volume)
            except (TypeError, ValueError):
                continue

        if not candidates:
            raise RuntimeError("No eligible USDT-margined altcoins found in tickers.")

        ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
        top_n = ranked[: max(1, NUM_ASSETS_TO_TRADE)]
        for sym, vol in top_n:
            logger.info("Volume scan %s: %.2f USDT (24h quote)", sym, vol)
        # NUM_ASSETS_TO_TRADE == 1 -> return the single top symbol.
        return top_n[0][0]

    def _daily_scan_due(self, now_ts: int) -> bool:
        """True when the current UTC wall clock is inside the 00:00 UTC window."""
        now = datetime.fromtimestamp(now_ts / 1000, tz=timezone.utc)
        window_s = DAILY_SCAN_WINDOW_SECONDS
        seconds_since_midnight = now.hour * 3600 + now.minute * 60 + now.second
        if seconds_since_midnight > window_s:
            return False
        if self.last_daily_scan_ts is not None:
            # Already scanned during this UTC day.
            last = datetime.fromtimestamp(
                self.last_daily_scan_ts / 1000, tz=timezone.utc)
            if last.date() == now.date():
                return False
        return True

    # ------------------------------------------------------- indicator math
    @staticmethod
    def _true_range(high: float, low: float, prev_close: float) -> float:
        return max(high - low, abs(high - prev_close), abs(low - prev_close))

    @staticmethod
    def _atr(candles, period: int = ATR_PERIOD) -> float:
        """Wilder-smoothed ATR over the most recent `period` closed candles.

        `candles` are CCXT OHLCV rows [ts, open, high, low, close, volume].
        The LAST candle is the in-progress candle and is excluded, so the ATR
        reflects only confirmed bars (no look-ahead).
        """
        closed = candles[:-1] if len(candles) > period else candles
        if len(closed) < period + 1:
            raise ValueError(
                f"Not enough candles for ATR: need {period + 1}, got {len(closed)}")
        trs = []
        prev_close = float(closed[0][4])
        for row in closed[1:]:
            high, low, close = float(row[2]), float(row[3]), float(row[4])
            trs.append(BinanceVolumeRotatorBot._true_range(high, low, prev_close))
            prev_close = close
        # Wilder smoothing: ATR_t = (ATR_{t-1}*(n-1) + TR_t) / n
        atr = sum(trs[:period]) / period
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period
        return atr

    @staticmethod
    def _donchian(candles, period: int = DONCHIAN_PERIOD):
        """Upper/lower Donchian bands over the last `period` CLOSED candles.

        Excludes the in-progress candle (the last element), so the bands are
        known before the breakout bar forms (no look-ahead bias).
        """
        closed = candles[:-1]
        if len(closed) < period:
            raise ValueError(
                f"Not enough candles for Donchian: need {period}, got {len(closed)}")
        highs = [float(row[2]) for row in closed[-period:]]
        lows = [float(row[3]) for row in closed[-period:]]
        return max(highs), min(lows)

    def _fetch_signal(self, symbol: str) -> SignalSnapshot:
        """Fetch candles and compute bands/ATR/signals for `symbol`."""
        candles = self.exchange.fetch_ohlcv(
            symbol, timeframe=TIMEFRAME, limit=CANDLE_HISTORY_LIMIT)
        if len(candles) < DONCHIAN_PERIOD + 1:
            raise RuntimeError(
                f"Only {len(candles)} candles for {symbol}; need at least "
                f"{DONCHIAN_PERIOD + 1}.")
        self.last_candles = candles  # store for chart

        price = float(candles[-1][4])  # latest close (in-progress bar)
        atr = self._atr(candles, ATR_PERIOD)
        upper, lower = self._donchian(candles, DONCHIAN_PERIOD)

        long_signal = price > upper
        short_signal = price < lower
        # Reversal: current price touches/crosses the OPPOSITE band.
        reversal_long = price < lower   # exit a long when lower touched
        reversal_short = price > upper  # exit a short when upper touched

        snap = SignalSnapshot(
            symbol=symbol,
            price=price,
            atr=atr,
            donchian_upper=upper,
            donchian_lower=lower,
            long_signal=long_signal,
            short_signal=short_signal,
            reversal_long=reversal_long,
            reversal_short=reversal_short,
        )
        self.last_snap = snap
        logger.info(
            "%s price=%.6f ATR=%.6f upper=%.6f lower=%.6f "
            "long=%s short=%s rev_long=%s rev_short=%s",
            symbol, price, atr, upper, lower,
            long_signal, short_signal, reversal_long, reversal_short,
        )
        return snap

    # -------------------------------------------------------- position sizing
    def _target_quantity(self, symbol: str, price: float, atr: float) -> float:
        """ATR-adjusted, leverage-capped base-asset quantity.

        Target_Notional = (Equity * RISK_FACTOR) * Price / ATR
        capped at Equity * LEVERAGE_CAP. Then notional / price = quantity.
        """
        equity = self._fetch_balance_usdt()
        target_notional = (equity * RISK_FACTOR) * price / atr
        cap_notional = equity * LEVERAGE_CAP
        if target_notional > cap_notional:
            logger.info(
                "Notional %.2f USDT exceeds leverage cap %.2f USDT; clamping.",
                target_notional, cap_notional,
            )
            target_notional = cap_notional

        quantity = target_notional / price
        quantity = self._clamp_amount(symbol, quantity)
        logger.info(
            "Sizing %s: equity=%.2f notional=%.2f price=%.6f qty=%.6f",
            symbol, equity, target_notional, price, quantity,
        )
        return quantity

    # ------------------------------------------------------------- execution
    def _set_leverage(self, symbol: str) -> None:
        """Align leverage to the cap (position sizing already clamps)."""
        try:
            self.exchange.set_leverage(int(LEVERAGE_CAP), symbol)
            logger.info("Leverage set to %sx for %s", int(LEVERAGE_CAP), symbol)
        except ccxt.ExchangeError as exc:
            logger.warning("set_leverage skipped for %s: %s", symbol, exc)

    def _maker_entry(self, symbol: str, side: str, amount: float,
                     price: float) -> Optional[dict]:
        """Place a postOnly LIMIT entry (maker). Fails instead of taking."""
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type="limit",
                side=side,
                amount=amount,
                price=self.exchange.price_to_precision(symbol, price),
                params={"postOnly": True, "reduceOnly": False},
            )
            logger.info("Maker entry placed: %s %s %s @ %s (id=%s)",
                        symbol, side, amount, price, order.get("id"))
            return order
        except ccxt.InsufficientFunds as exc:
            logger.error("Insufficient margin for %s %s %s: %s",
                         side, amount, symbol, exc)
            return None
        except ccxt.ExchangeError as exc:
            logger.error("Maker entry failed (%s %s): %s", side, symbol, exc)
            return None

    def _cancel_order(self, order_id: Optional[str], symbol: str) -> None:
        if not order_id:
            return
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info("Cancelled order %s", order_id)
        except ccxt.ExchangeError as exc:
            logger.warning("Cancel order %s failed: %s", order_id, exc)

    def _place_stop_loss(self, symbol: str, side: str, amount: float,
                         stop_price: float) -> Optional[str]:
        """STOP_MARKET trailing stop opposite to the position side."""
        # For a long, stop is a SELL stop; for a short, stop is a BUY stop.
        stop_side = "sell" if side == "buy" else "buy"
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type="STOP_MARKET",
                side=stop_side,
                amount=amount,
                params={
                    "stopPrice": self.exchange.price_to_precision(
                        symbol, stop_price),
                    "reduceOnly": True,
                    "closePosition": False,
                },
            )
            oid = str(order.get("id"))
            logger.info("Trailing STOP_MARKET placed %s @ %.6f (id=%s)",
                        stop_side, stop_price, oid)
            return oid
        except ccxt.ExchangeError as exc:
            logger.error("STOP_MARKET placement failed: %s", exc)
            return None
    def _update_trailing_stop(self, symbol: str, side: str, amount: float,
                              current_atr: float, current_price: float) -> None:
        """Move the stop to 4.0x ATR from current price when favorable.

        Long  -> stop = price - 4*ATR, only RAISES the stop.
        Short -> stop = price + 4*ATR, only LOWERS the stop.
        """
        if side == "buy":  # long position
            new_stop = current_price - ATR_STOP_MULTIPLIER * current_atr
            if new_stop <= self.trailing_stop_price:
                return
        else:  # short position
            new_stop = current_price + ATR_STOP_MULTIPLIER * current_atr
            if new_stop >= self.trailing_stop_price:
                return

        # Replace the outstanding stop: cancel old, place new.
        self._cancel_order(self.trailing_stop_order_id, symbol)
        self.trailing_stop_price = new_stop
        self.trailing_stop_order_id = self._place_stop_loss(
            symbol, side, amount, new_stop)

    def _exit_market(self, symbol: str, side: str, amount: float) -> bool:
        """Close the position via MARKET order (taker)."""
        close_side = "sell" if side == "buy" else "buy"
        try:
            self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=close_side,
                amount=amount,
                params={"reduceOnly": True},
            )
            logger.info("MARKET exit executed: %s %s %s", close_side, amount, symbol)
            return True
        except ccxt.ExchangeError as exc:
            logger.error("MARKET exit failed (%s %s): %s", close_side, symbol, exc)
            return False

    def _clear_position_state(self) -> None:
        self.position_side = None
        self.position_amount = 0.0
        self.entry_price = 0.0
        self.trailing_stop_price = 0.0
        self.trailing_stop_order_id = None

    # ------------------------------------------------------- rich dashboard / chart
    def _render_dashboard(self, snap: Optional[SignalSnapshot], balance: float) -> Panel:
        """Build a rich Panel with current bot status."""
        if snap is None:
            return Panel("Waiting for data...", title="Bot Status", border_style="yellow")

        # Header
        header = Table.grid(expand=True)
        header.add_column(justify="left", style="bold cyan")
        header.add_column(justify="right", style="bold green")
        header.add_row(f"Symbol: {snap.symbol}", f"Balance: {balance:.2f} USDT")

        # Strategy table
        table = Table(box=box.ROUNDED, title_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Price", f"{snap.price:.6f}")
        table.add_row("Upper Band (Long trigger)", f"{snap.donchian_upper:.6f}")
        table.add_row("Lower Band (Short trigger)", f"{snap.donchian_lower:.6f}")
        table.add_row("ATR", f"{snap.atr:.6f}")

        # Distance to trigger
        band_width = snap.donchian_upper - snap.donchian_lower
        if band_width > 0:
            pos_in_band = (snap.price - snap.donchian_lower) / band_width
            bar_len = 20
            filled = int(pos_in_band * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            table.add_row("Band Position", f"{bar} {pos_in_band:.1%}")

        # Signals
        sig = ""
        if snap.long_signal:
            sig += "🔺 LONG "
        if snap.short_signal:
            sig += "🔻 SHORT "
        if not sig:
            sig = "—"
        table.add_row("Signal", sig)

        # Position
        if self.position_side is not None:
            pnl = 0.0
            if self.position_side == "long":
                pnl = (snap.price - self.entry_price) * self.position_amount
            else:
                pnl = (self.entry_price - snap.price) * self.position_amount
            pnl_str = f"{pnl:+.2f} USDT"
            table.add_row("Position", f"{self.position_side.upper()} {self.position_amount:.4f}")
            table.add_row("Entry", f"{self.entry_price:.6f}")
            table.add_row("Trailing Stop", f"{self.trailing_stop_price:.6f}")
            table.add_row("Unrealized P&L", pnl_str)
        else:
            table.add_row("Position", "FLAT")

        layout = Layout()
        layout.split(
            Layout(header, size=3),
            Layout(table),
        )
        return Panel(layout, title=f"Binance Volume Rotator  |  {datetime.now():%H:%M:%S}", border_style="blue")

    def _generate_chart(self, snap: SignalSnapshot) -> None:
        """Plot candles + bands + price + stop and save to live_chart.png."""
        if not self.last_candles or self.last_candles is None:
            return
        candles = self.last_candles
        # Prepare data
        times = [datetime.fromtimestamp(c[0]/1000) for c in candles]
        opens = [c[1] for c in candles]
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        closes = [c[4] for c in candles]

        fig, ax = plt.subplots(figsize=(12, 6))
        # Plot candlesticks (simple line for speed, but we can do better)
        ax.plot(times, closes, label='Close', color='black', linewidth=1)
        # Add Donchian bands
        # Need to compute bands on the fly from self.last_candles? We have snap with upper/lower but they are based on last 120 closed candles.
        # We can compute upper/lower for all points for visual; we only have current snap values, but we can compute for each candle.
        # For simplicity, we'll plot horizontal lines for current bands.
        ax.axhline(y=snap.donchian_upper, color='green', linestyle='--', label='Upper Band (Long)')
        ax.axhline(y=snap.donchian_lower, color='red', linestyle='--', label='Lower Band (Short)')
        # Current price
        ax.axhline(y=snap.price, color='blue', linestyle='-', label='Current Price')
        # Trailing stop if in position
        if self.position_side is not None and self.trailing_stop_price > 0:
            ax.axhline(y=self.trailing_stop_price, color='orange', linestyle='-.', label='Trailing Stop')
        ax.legend()
        ax.set_title(f"{snap.symbol}  |  {TIMEFRAME}  |  Donchian {DONCHIAN_PERIOD}")
        ax.grid(True, alpha=0.3)
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig("live_chart.png", dpi=100)
        plt.close(fig)

    def _enter(self, symbol: str, side: str, snap: SignalSnapshot) -> None:
        """Size, set leverage, place maker entry, then arm the stop."""
        quantity = self._target_quantity(symbol, snap.price, snap.atr)
        if quantity <= 0:
            logger.warning("Skipping entry: computed quantity <= 0.")
            return

        self._set_leverage(symbol)
        order = self._maker_entry(symbol, side, quantity, snap.price)
        if order is None:
            return

        # Record intended state; stop armed immediately. A production build
        # should poll the fill status before trusting the entry; for maker
        # postOnly the order either fills (maker) or rests unfilled.
        self.position_side = "long" if side == "buy" else "short"
        self.position_amount = float(order.get("amount") or quantity)
        self.entry_price = snap.price
        if side == "buy":
            self.trailing_stop_price = snap.price - ATR_STOP_MULTIPLIER * snap.atr
        else:
            self.trailing_stop_price = snap.price + ATR_STOP_MULTIPLIER * snap.atr
        self.trailing_stop_order_id = self._place_stop_loss(
            symbol, side, self.position_amount, self.trailing_stop_price)
        logger.info("Entered %s %s @ %.6f qty=%.6f stop=%.6f",
                    side, symbol, snap.price, self.position_amount,
                    self.trailing_stop_price)

    # ------------------------------------------------------------- main loop
    def run(self) -> None:
        """Main event loop: daily rotation + candle polling, with rich live dashboard."""
        logger.info("Bot starting. Demo Trading=%s RISK_FACTOR=%.2f%% "
                    "LEVERAGE_CAP=%.1fx DONCHIAN=%d ATR=%d stop=%.1fx TF=%s",
                    USE_TESTNET, RISK_FACTOR * 100, LEVERAGE_CAP,
                    DONCHIAN_PERIOD, ATR_PERIOD, ATR_STOP_MULTIPLIER,
                    TIMEFRAME)

        with Live(self._render_dashboard(None, 0.0), refresh_per_second=1) as live:
            while True:
                try:
                    now_ts = self.exchange.milliseconds()

                    # 1. Daily asset rotation at ~00:00 UTC.
                    if self.active_symbol is None or self._daily_scan_due(now_ts):
                        new_symbol = self.get_top_volume_altcoin()
                        if new_symbol != self.active_symbol:
                            # Rotate: close any existing position before switching.
                            if self.position_side is not None:
                                self._exit_market(
                                    self.active_symbol,
                                    "buy" if self.position_side == "long" else "sell",
                                    self.position_amount,
                                )
                                self._cancel_order(
                                    self.trailing_stop_order_id, self.active_symbol)
                                self._clear_position_state()
                            self.active_symbol = new_symbol
                            self.last_daily_scan_ts = now_ts
                            logger.info("Active symbol rotated to %s", new_symbol)

                    if self.active_symbol is None:
                        live.update(self._render_dashboard(None, self.last_balance))
                        time.sleep(POLL_SECONDS)
                        continue

                    symbol = self.active_symbol
                    snap = self._fetch_signal(symbol)
                    self.last_balance = self._fetch_balance_usdt()

                    # 2. Reversal exit check (opposite band touch).
                    if self.position_side == "long" and snap.reversal_long:
                        logger.info("Reversal exit (lower band touched) -> close long.")
                        self._exit_market(symbol, "buy", self.position_amount)
                        self._cancel_order(self.trailing_stop_order_id, symbol)
                        self._clear_position_state()
                    elif self.position_side == "short" and snap.reversal_short:
                        logger.info("Reversal exit (upper band touched) -> close short.")
                        self._exit_market(symbol, "sell", self.position_amount)
                        self._cancel_order(self.trailing_stop_order_id, symbol)
                        self._clear_position_state()

                    # 3. Trailing stop update while in position.
                    if self.position_side is not None:
                        self._update_trailing_stop(
                            symbol,
                            "buy" if self.position_side == "long" else "sell",
                            self.position_amount,
                            snap.atr,
                            snap.price,
                        )

                    # 4. Entry signals only when flat.
                    if self.position_side is None:
                        if snap.long_signal:
                            self._enter(symbol, "buy", snap)
                        elif snap.short_signal:
                            self._enter(symbol, "sell", snap)

                    # Update dashboard and generate chart
                    live.update(self._render_dashboard(snap, self.last_balance))
                    self._generate_chart(snap)

                except ccxt.NetworkError as exc:
                    logger.warning("Network error: %s", exc)
                    live.update(self._render_dashboard(self.last_snap, self.last_balance))
                except ccxt.RateLimitExceeded as exc:
                    logger.warning("Rate limit hit: %s", exc)
                    live.update(self._render_dashboard(self.last_snap, self.last_balance))
                except ccxt.InsufficientFunds as exc:
                    logger.error("Margin error: %s", exc)
                    live.update(self._render_dashboard(self.last_snap, self.last_balance))
                except ccxt.ExchangeError as exc:
                    logger.error("Exchange error: %s", exc)
                    live.update(self._render_dashboard(self.last_snap, self.last_balance))
                except Exception as exc:  # noqa: BLE001 - keep the loop alive
                    logger.exception("Unexpected error: %s", exc)
                    live.update(self._render_dashboard(self.last_snap, self.last_balance))

                time.sleep(POLL_SECONDS)


def main() -> int:
    bot = BinanceVolumeRotatorBot()
    bot.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
