"""Volatility-Ranked Asymmetric Portfolio Backtester.

Aronson EBTA: concentrate capital on the Top-3 most volatile assets
(14-ATR / Close normalized volatility over a trailing 672-bar window) and run
an asymmetric momentum system:

  Trend   : EMA-200 on close (long above, short below)
  Entry   : Donchian-20 breakout + volume > 1.5x SMA-20
  Risk    : initial SL 2.5x ATR, TP 5.0x ATR (1:2 R:R)
            + dynamic ATR trailing stop (only moves in trade direction)
  NO symmetric ADX exit. Trailing stop is the primary exit.

Friction: 0.05% taker + 0.02% slippage each side -> 0.14% round-trip.
Segments : Training 0-4999 | Testing 5000-7499 | Validation 7500-9999.

Usage:  python scratch/scalping_volatility_portfolio.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError as exc:
    print(f"ccxt missing: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("volatility_portfolio")

# --------------------------------------------------------------------- cfg
TIMEFRAME = "15m"
TARGET_CANDLES = 10000
INITIAL_CAPITAL = 100.0     # per-slot allocation (USDT)
PAGE_LIMIT = 1000
PAGE_SLEEP = 0.5

SYMBOLS = [
    "SOL/USDT:USDT",
    "DOGE/USDT:USDT",
    "XRP/USDT:USDT",
    "ADA/USDT:USDT",
    "LINK/USDT:USDT",
    "AVAX/USDT:USDT",
    "DOT/USDT:USDT",
    "POL/USDT:USDT",
    "LTC/USDT:USDT",
    "1000SHIB/USDT:USDT",
]

# Friction model (Taker execution on both sides).
TAKER_FEE = 0.0005          # 0.05% on entry + 0.05% on exit
SLIPPAGE = 0.0002           # 0.02% adverse fill on entry + 0.02% on exit
FRICTION_RT_PCT = (TAKER_FEE + SLIPPAGE) * 2 * 100   # 0.14% round-trip

# Strategy parameters (asymmetric: trend + breakout + volume, no ADX exit).
EMA_PERIOD = 200
DONCHIAN_PERIOD = 20
ADX_PERIOD = 14
ADX_MIN = 25.0              # used only by the baseline binary/tri-state engines
VOL_MA_PERIOD = 20
VOL_MULT = 1.5              # volume must exceed 1.5x its 20-bar average
ATR_PERIOD = 14
ATR_SL = 2.5                # stop distance in ATR units
ATR_TP = 5.0                # target distance in ATR units (1:2.0 R:R)

VOL_RANK_WINDOW = 672       # trailing candles used for volatility ranking
TOP_N = 3                   # trade the Top-3 most volatile assets

SLEEP_BETWEEN_ASSETS = 0.75

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "out")
OUT_PNG = os.path.join(OUT_DIR, "volatility_portfolio_results.png")

# Strict chronological partition boundaries (0-based row positions).
TRAIN_END = 5000
TEST_END = 7500
SEGMENTS: List = [
    ("Training", 0, TRAIN_END),
    ("Testing", TRAIN_END, TEST_END),
    ("Validation", TEST_END, TARGET_CANDLES),
]
SEGMENT_NAMES = [s[0] for s in SEGMENTS]

# -------------------------------------------------------------- indicators
def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int) -> pd.Series:
    """Wilder-smoothed Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    """Wilder-smoothed Average Directional Index (ADX)."""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(0.0, index=close.index)
    minus_dm = pd.Series(0.0, index=close.index)
    mask_plus = (up_move > down_move) & (up_move > 0)
    mask_minus = (down_move > up_move) & (down_move > 0)
    plus_dm[mask_plus] = up_move[mask_plus]
    minus_dm[mask_minus] = down_move[mask_minus]

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1.0 / period, adjust=False).mean()

    plus_di = 100.0 * plus_dm.ewm(
        alpha=1.0 / period, adjust=False).mean() / atr_.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(
        alpha=1.0 / period, adjust=False).mean() / atr_.replace(0.0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / \
        (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


def donchian_bands(high: pd.Series, low: pd.Series,
                   period: int):
    """Return (upper, lower) Donchian bands over the PRIOR `period` bars.

    Uses shift(1) so the current bar is excluded -> zero look-ahead.
    """
    upper = high.shift(1).rolling(period, min_periods=period).max()
    lower = low.shift(1).rolling(period, min_periods=period).min()
    return upper, lower


# ---------------------------------------------------------------- data io
def create_exchange() -> ccxt.Exchange:
    return ccxt.binanceusdm({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })


def tf_to_ms(timeframe: str) -> int:
    """Convert a CCXT timeframe string to milliseconds."""
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    if unit not in units:
        raise ValueError(f"Unsupported timeframe unit: {timeframe!r}")
    return value * units[unit]


def fetch_10k_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str,
                    target_candles: int = TARGET_CANDLES) -> pd.DataFrame:
    """Paginate Binance USD-M OHLCV backward until `target_candles` unique bars."""
    step_ms = tf_to_ms(timeframe)
    rows: Dict[int, list] = {}
    since = None
    iterations = 0
    max_iterations = (target_candles // PAGE_LIMIT) + 5

    while len(rows) < target_candles and iterations < max_iterations:
        iterations += 1
        try:
            if since is None:
                batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe,
                                             limit=PAGE_LIMIT)
            else:
                batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe,
                                             since=since, limit=PAGE_LIMIT)
        except Exception as exc:
            logger.warning("fetch_10k_ohlcv page %d failed for %s: %s",
                           iterations, symbol, exc)
            break

        if not batch:
            break

        for row in batch:
            rows[int(row[0])] = row

        earliest = int(batch[0][0])
        # Move the inclusive startTime one full page BEFORE the current oldest
        # bar, so the next page's last candle lands exactly one step older.
        since = earliest - step_ms * PAGE_LIMIT

        if len(rows) >= target_candles:
            break
        time.sleep(PAGE_SLEEP)

    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"])

    ordered = [rows[ts] for ts in sorted(rows)]
    df = pd.DataFrame(
        ordered, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df = df.set_index("timestamp")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df.iloc[:target_candles]


# ---------------------------------------------------- volatility ranking
def rank_assets_by_vol(assets: Dict[str, pd.DataFrame],
                       segment_start: int) -> List:
    """Rank assets by normalized ATR (ATR/Close) over the trailing window.

    Uses candles strictly BEFORE the segment start (no OOS look-ahead). For the
    Training segment (start 0) the first VOL_RANK_WINDOW bars serve as warm-up.
    """
    scores: Dict[str, float] = {}
    for asset, df in assets.items():
        if segment_start == 0:
            win = df.iloc[0:VOL_RANK_WINDOW]
        else:
            win = df.iloc[segment_start - VOL_RANK_WINDOW:segment_start]
        if len(win) < 100:
            scores[asset] = -np.inf
            continue
        atr = compute_atr(win["high"], win["low"], win["close"], ATR_PERIOD)
        nvol = (atr / win["close"]) * 100.0
        nvol = nvol.dropna()
        scores[asset] = float(nvol.tail(200).mean()) if len(nvol) else -np.inf
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

# --------------------------------------------------------------- engines
def _entry_frame(df: pd.DataFrame) -> pd.Series:
    """Indicator block shared by all engines; raw entry signal (no ADX gate):
    +1 long breakout, -1 short breakout."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema = close.ewm(span=EMA_PERIOD, adjust=False).mean()
    upper, lower = donchian_bands(high, low, DONCHIAN_PERIOD)
    vol_ma = df["volume"].rolling(VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean()

    trend_long = close > ema
    trend_short = close < ema
    breakout_up = close > upper
    breakout_down = close < lower
    vol_ok = df["volume"] > (VOL_MULT * vol_ma)

    long_sig = trend_long & breakout_up & vol_ok
    short_sig = trend_short & breakout_down & vol_ok
    valid = ema.notna() & upper.notna() & vol_ma.notna()

    entry_signal = pd.Series(0, index=df.index, dtype="int64")
    entry_signal[long_sig & valid] = 1
    entry_signal[short_sig & valid] = -1
    return entry_signal


def run_asymmetric(df: pd.DataFrame) -> Dict:
    """Asymmetric engine: no ADX filter, no symmetric ADX exit.

    Entry at open t+1 from close-t signal. Initial SL 2.5 ATR / TP 5.0 ATR.
    Once favorable, trail SL at extreme -/+ 2.5 ATR (only moves trade-ward).
    """
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]

    atr = compute_atr(high, low, close, ATR_PERIOD)
    entry_signal = _entry_frame(df)

    equity = INITIAL_CAPITAL
    position = 0           # 0 flat | 1 long | -1 short
    size = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    sl = tp = 0.0
    extreme = 0.0
    entry_equity = INITIAL_CAPITAL

    trades: List[Dict] = []
    eq_rows: List = []

    n = len(df)
    for i in range(n):
        ts = df.index[i]
        o = open_.iloc[i]
        h = high.iloc[i]
        l = low.iloc[i]

        # 1) Entry at this bar's open from previous bar's signal.
        if position == 0 and i > 0:
            sig = int(entry_signal.iloc[i - 1])
            a = atr.iloc[i - 1]
            if sig != 0 and np.isfinite(a) and a > 0:
                if sig == 1:
                    fill = o * (1.0 + SLIPPAGE)
                else:
                    fill = o * (1.0 - SLIPPAGE)
                entry_equity = equity
                size = entry_equity / fill
                equity = entry_equity - entry_equity * TAKER_FEE
                entry_price = fill
                entry_atr = a
                if sig == 1:
                    sl = entry_price - ATR_SL * a
                    tp = entry_price + ATR_TP * a
                    extreme = fill
                else:
                    sl = entry_price + ATR_SL * a
                    tp = entry_price - ATR_TP * a
                    extreme = fill
                position = sig

        # 2) Update trailing stop (only moves in the trade direction).
        if position == 1:
            a_prev = atr.iloc[i - 1] if i > 0 else entry_atr
            if np.isfinite(a_prev) and a_prev > 0:
                extreme = max(extreme, h)
                new_sl = extreme - ATR_SL * a_prev
                sl = max(sl, new_sl)
        elif position == -1:
            a_prev = atr.iloc[i - 1] if i > 0 else entry_atr
            if np.isfinite(a_prev) and a_prev > 0:
                extreme = min(extreme, l)
                new_sl = extreme + ATR_SL * a_prev
                sl = min(sl, new_sl)

        # 3) Exit checks (trailing stop checked first -> conservative).
        exit_fill = None
        exit_reason = ""
        if position == 1:
            if l <= sl:
                base = min(o, sl)
                exit_fill = base * (1.0 - SLIPPAGE)
                exit_reason = "TRAIL"
            elif h >= tp:
                exit_fill = tp * (1.0 - SLIPPAGE)
                exit_reason = "TP"
        elif position == -1:
            if h >= sl:
                base = max(o, sl)
                exit_fill = base * (1.0 + SLIPPAGE)
                exit_reason = "TRAIL"
            elif l <= tp:
                exit_fill = tp * (1.0 + SLIPPAGE)
                exit_reason = "TP"

        if exit_fill is not None:
            proceeds = size * exit_fill
            exit_fee = proceeds * TAKER_FEE
            equity = proceeds - exit_fee
            trades.append({
                "pnl": equity - entry_equity,
                "return_pct": (equity / entry_equity - 1.0) * 100.0,
                "side": position,
                "reason": exit_reason,
            })
            position = 0
            size = 0.0

        eq_rows.append((ts, equity))

    # 4) Force-close any open position at the final close.
    if position != 0:
        last_close = close.iloc[-1]
        if position == 1:
            exit_fill = last_close * (1.0 - SLIPPAGE)
        else:
            exit_fill = last_close * (1.0 + SLIPPAGE)
        proceeds = size * exit_fill
        exit_fee = proceeds * TAKER_FEE
        equity = proceeds - exit_fee
        trades.append({
            "pnl": equity - entry_equity,
            "return_pct": (equity / entry_equity - 1.0) * 100.0,
            "side": position,
            "reason": "EOD",
        })
        eq_rows[-1] = (df.index[-1], equity)

    equity_series = pd.DataFrame(
        eq_rows, columns=["timestamp", "equity"]).set_index("timestamp")["equity"]
    return {
        "equity": equity_series,
        "growth": equity_series / INITIAL_CAPITAL,
        "close_norm": close / close.iloc[0],
        "trades": trades,
    }

def run_binary(df: pd.DataFrame) -> Dict:
    """Baseline binary engine (previous script): EMA200 + Donchian20 + volume
    + ADX > 25 entry, fixed SL 2.5 ATR / TP 5.0 ATR, no trailing."""
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]

    ema = close.ewm(span=EMA_PERIOD, adjust=False).mean()
    atr = compute_atr(high, low, close, ATR_PERIOD)
    adx = compute_adx(high, low, close, ADX_PERIOD)
    upper, lower = donchian_bands(high, low, DONCHIAN_PERIOD)
    vol_ma = df["volume"].rolling(VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean()

    trend_long = close > ema
    trend_short = close < ema
    breakout_up = close > upper
    breakout_down = close < lower
    vol_ok = df["volume"] > (VOL_MULT * vol_ma)
    adx_ok = adx > ADX_MIN

    long_sig = trend_long & breakout_up & vol_ok & adx_ok
    short_sig = trend_short & breakout_down & vol_ok & adx_ok
    valid = ema.notna() & atr.notna() & adx.notna() & upper.notna() & vol_ma.notna()

    entry_signal = pd.Series(0, index=df.index, dtype="int64")
    entry_signal[long_sig & valid] = 1
    entry_signal[short_sig & valid] = -1

    equity = INITIAL_CAPITAL
    position = 0
    size = 0.0
    entry_price = 0.0
    sl = tp = 0.0
    entry_equity = INITIAL_CAPITAL

    trades: List[Dict] = []
    eq_rows: List = []

    n = len(df)
    for i in range(n):
        ts = df.index[i]
        o = open_.iloc[i]
        h = high.iloc[i]
        l = low.iloc[i]

        if position == 0 and i > 0:
            sig = int(entry_signal.iloc[i - 1])
            a = atr.iloc[i - 1]
            if sig != 0 and np.isfinite(a) and a > 0:
                if sig == 1:
                    fill = o * (1.0 + SLIPPAGE)
                else:
                    fill = o * (1.0 - SLIPPAGE)
                entry_equity = equity
                size = entry_equity / fill
                equity = entry_equity - entry_equity * TAKER_FEE
                entry_price = fill
                if sig == 1:
                    sl = entry_price - ATR_SL * a
                    tp = entry_price + ATR_TP * a
                else:
                    sl = entry_price + ATR_SL * a
                    tp = entry_price - ATR_TP * a
                position = sig

        exit_fill = None
        exit_reason = ""
        if position == 1:
            if l <= sl:
                base = min(o, sl)
                exit_fill = base * (1.0 - SLIPPAGE)
                exit_reason = "SL"
            elif h >= tp:
                exit_fill = tp * (1.0 - SLIPPAGE)
                exit_reason = "TP"
        elif position == -1:
            if h >= sl:
                base = max(o, sl)
                exit_fill = base * (1.0 + SLIPPAGE)
                exit_reason = "SL"
            elif l <= tp:
                exit_fill = tp * (1.0 + SLIPPAGE)
                exit_reason = "TP"

        if exit_fill is not None:
            proceeds = size * exit_fill
            exit_fee = proceeds * TAKER_FEE
            equity = proceeds - exit_fee
            trades.append({
                "pnl": equity - entry_equity,
                "return_pct": (equity / entry_equity - 1.0) * 100.0,
                "side": position,
                "reason": exit_reason,
            })
            position = 0
            size = 0.0

        eq_rows.append((ts, equity))

    if position != 0:
        last_close = close.iloc[-1]
        if position == 1:
            exit_fill = last_close * (1.0 - SLIPPAGE)
        else:
            exit_fill = last_close * (1.0 + SLIPPAGE)
        proceeds = size * exit_fill
        exit_fee = proceeds * TAKER_FEE
        equity = proceeds - exit_fee
        trades.append({
            "pnl": equity - entry_equity,
            "return_pct": (equity / entry_equity - 1.0) * 100.0,
            "side": position,
            "reason": "EOD",
        })
        eq_rows[-1] = (df.index[-1], equity)

    equity_series = pd.DataFrame(
        eq_rows, columns=["timestamp", "equity"]).set_index("timestamp")["equity"]
    return {
        "equity": equity_series,
        "growth": equity_series / INITIAL_CAPITAL,
        "close_norm": close / close.iloc[0],
        "trades": trades,
    }

def run_tristate(df: pd.DataFrame) -> Dict:
    """Baseline tri-state engine (previous script): same as binary plus an
    ADX regime guard (ADX < 25 -> cash, instant flat, no re-entry)."""
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]

    ema = close.ewm(span=EMA_PERIOD, adjust=False).mean()
    atr = compute_atr(high, low, close, ATR_PERIOD)
    adx = compute_adx(high, low, close, ADX_PERIOD)
    upper, lower = donchian_bands(high, low, DONCHIAN_PERIOD)
    vol_ma = df["volume"].rolling(VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean()

    trend_long = close > ema
    trend_short = close < ema
    breakout_up = close > upper
    breakout_down = close < lower
    vol_ok = df["volume"] > (VOL_MULT * vol_ma)
    adx_ok = adx > ADX_MIN

    long_sig = trend_long & breakout_up & vol_ok & adx_ok
    short_sig = trend_short & breakout_down & vol_ok & adx_ok
    valid = ema.notna() & atr.notna() & adx.notna() & upper.notna() & vol_ma.notna()

    entry_signal = pd.Series(0, index=df.index, dtype="int64")
    entry_signal[long_sig & valid] = 1
    entry_signal[short_sig & valid] = -1

    equity = INITIAL_CAPITAL
    position = 0
    size = 0.0
    sl = tp = 0.0
    entry_equity = INITIAL_CAPITAL
    flat_count = 0

    trades: List[Dict] = []
    eq_rows: List = []

    n = len(df)
    for i in range(n):
        ts = df.index[i]
        o = open_.iloc[i]
        h = high.iloc[i]
        l = low.iloc[i]
        adx_i = float(adx.iloc[i])

        # 0) Regime guard: choppy -> instant flat at bar open.
        if position != 0 and np.isfinite(adx_i) and adx_i < ADX_MIN:
            if position == 1:
                exit_fill = o * (1.0 - SLIPPAGE)
            else:
                exit_fill = o * (1.0 + SLIPPAGE)
            proceeds = size * exit_fill
            exit_fee = proceeds * TAKER_FEE
            equity = proceeds - exit_fee
            trades.append({
                "pnl": equity - entry_equity,
                "return_pct": (equity / entry_equity - 1.0) * 100.0,
                "side": position,
                "reason": "REGIME",
            })
            position = 0
            size = 0.0
            flat_count += 1

        # 1) Entry at this bar's open from previous bar's signal.
        if position == 0 and i > 0:
            regime_ok = np.isfinite(adx_i) and adx_i >= ADX_MIN
            sig = int(entry_signal.iloc[i - 1])
            a = atr.iloc[i - 1]
            if regime_ok and sig != 0 and np.isfinite(a) and a > 0:
                if sig == 1:
                    fill = o * (1.0 + SLIPPAGE)
                else:
                    fill = o * (1.0 - SLIPPAGE)
                entry_equity = equity
                size = entry_equity / fill
                equity = entry_equity - entry_equity * TAKER_FEE
                if sig == 1:
                    sl = fill - ATR_SL * a
                    tp = fill + ATR_TP * a
                else:
                    sl = fill + ATR_SL * a
                    tp = fill - ATR_TP * a
                position = sig

        # 2) Exit checks.
        exit_fill = None
        exit_reason = ""
        if position == 1:
            if l <= sl:
                base = min(o, sl)
                exit_fill = base * (1.0 - SLIPPAGE)
                exit_reason = "SL"
            elif h >= tp:
                exit_fill = tp * (1.0 - SLIPPAGE)
                exit_reason = "TP"
        elif position == -1:
            if h >= sl:
                base = max(o, sl)
                exit_fill = base * (1.0 + SLIPPAGE)
                exit_reason = "SL"
            elif l <= tp:
                exit_fill = tp * (1.0 + SLIPPAGE)
                exit_reason = "TP"

        if exit_fill is not None:
            proceeds = size * exit_fill
            exit_fee = proceeds * TAKER_FEE
            equity = proceeds - exit_fee
            trades.append({
                "pnl": equity - entry_equity,
                "return_pct": (equity / entry_equity - 1.0) * 100.0,
                "side": position,
                "reason": exit_reason,
            })
            position = 0
            size = 0.0

        eq_rows.append((ts, equity))

    if position != 0:
        last_close = close.iloc[-1]
        if position == 1:
            exit_fill = last_close * (1.0 - SLIPPAGE)
        else:
            exit_fill = last_close * (1.0 + SLIPPAGE)
        proceeds = size * exit_fill
        exit_fee = proceeds * TAKER_FEE
        equity = proceeds - exit_fee
        trades.append({
            "pnl": equity - entry_equity,
            "return_pct": (equity / entry_equity - 1.0) * 100.0,
            "side": position,
            "reason": "EOD",
        })
        eq_rows[-1] = (df.index[-1], equity)

    equity_series = pd.DataFrame(
        eq_rows, columns=["timestamp", "equity"]).set_index("timestamp")["equity"]
    return {
        "equity": equity_series,
        "growth": equity_series / INITIAL_CAPITAL,
        "close_norm": close / close.iloc[0],
        "trades": trades,
        "regime_flats": flat_count,
    }

# ---------------------------------------------------------------- helpers
def max_drawdown_pct(equity_series: pd.Series) -> float:
    if equity_series is None or len(equity_series) == 0:
        return 0.0
    cummax = equity_series.cummax()
    dd = (equity_series / cummax - 1.0) * 100.0
    return float(dd.min())


def summarize(equity_series: pd.Series, start_capital: float,
              trades: Optional[List[Dict]] = None) -> Dict:
    if equity_series is None or len(equity_series) == 0:
        return {"final_equity": start_capital, "net_return": 0.0,
                "max_drawdown": 0.0, "trades": 0, "win_rate": 0.0,
                "profit_factor": 0.0, "avg_trade_pct": 0.0}
    final = float(equity_series.iloc[-1])
    net_return = (final / start_capital - 1.0) * 100.0
    max_dd = max_drawdown_pct(equity_series)
    trades = trades or []
    n = len(trades)
    if n == 0:
        return {"final_equity": final, "net_return": net_return,
                "max_drawdown": max_dd, "trades": 0, "win_rate": 0.0,
                "profit_factor": 0.0, "avg_trade_pct": 0.0}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = float(sum(wins))
    gross_loss = float(abs(sum(losses)))
    win_rate = len(wins) / n * 100.0
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    elif gross_win > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0
    avg_trade_pct = float(np.mean([t["return_pct"] for t in trades]))

    return {"final_equity": final, "net_return": net_return,
            "max_drawdown": max_dd, "trades": n, "win_rate": win_rate,
            "profit_factor": profit_factor, "avg_trade_pct": avg_trade_pct}


def fmt_pf(pf: float) -> str:
    return "inf" if np.isinf(pf) else f"{pf:.2f}"


def combine_equity_growth(series_list: List[pd.Series]) -> pd.Series:
    """Equal-weight combination of per-slot equity series, normalized to $100."""
    if not series_list:
        return pd.Series(dtype=float)
    idx = None
    for s in series_list:
        idx = s.index if idx is None else idx.union(s.index)
    idx = idx.sort_values()
    aligned = pd.DataFrame({i: s.reindex(idx).ffill()
                            for i, s in enumerate(series_list)})
    total = aligned.sum(axis=1)
    start_total = INITIAL_CAPITAL * len(series_list)
    return total / start_total * INITIAL_CAPITAL


def basket_equity(growth_map: Dict[str, pd.Series]) -> pd.Series:
    """Equal-weight basket of per-asset growth series, normalized to $100."""
    if not growth_map:
        return pd.Series(dtype=float)
    idx = None
    for s in growth_map.values():
        idx = s.index if idx is None else idx.union(s.index)
    idx = idx.sort_values()
    aligned = pd.DataFrame({a: g.reindex(idx).ffill()
                            for a, g in growth_map.items()})
    return aligned.mean(axis=1) * INITIAL_CAPITAL

# ------------------------------------------------------------- main driver
def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    exchange = create_exchange()

    print(f"Volatility-Ranked Asymmetric Portfolio | {TIMEFRAME} | "
          f"{TARGET_CANDLES} candles/asset")
    print(f"Friction round-trip: {FRICTION_RT_PCT:.2f}% "
          f"(Taker {TAKER_FEE * 100:.2f}%/side + Slippage {SLIPPAGE * 100:.2f}%/side)")
    print(f"Top-{TOP_N} selection window: {VOL_RANK_WINDOW} candles "
          f"(ATR-{ATR_PERIOD}/Close)")
    print(f"Segments: Training 0-{TRAIN_END - 1} | "
          f"Testing {TRAIN_END}-{TEST_END - 1} | "
          f"Validation {TEST_END}-{TARGET_CANDLES - 1}\n")

    # ------------------------------------------------------------ fetch data
    assets_data: Dict[str, pd.DataFrame] = {}
    for idx, symbol in enumerate(SYMBOLS, start=1):
        asset = symbol.split("/")[0]
        print(f"[{idx}/{len(SYMBOLS)}] Fetching {symbol} ...")
        try:
            df = fetch_10k_ohlcv(exchange, symbol, TIMEFRAME, TARGET_CANDLES)
        except Exception as exc:
            logger.warning("SKIP %s: %s", symbol, exc)
            continue
        if len(df) < TARGET_CANDLES:
            logger.warning("SKIP %s: only %d bars (need %d)",
                           symbol, len(df), TARGET_CANDLES)
            continue
        assets_data[asset] = df.iloc[:TARGET_CANDLES]
        time.sleep(SLEEP_BETWEEN_ASSETS)

    if len(assets_data) < TOP_N:
        print("Not enough assets - aborting.", file=sys.stderr)
        return 1

    # -------------------------------------------------- top3 selection
    top_per_seg: Dict[str, List] = {}
    print("\n" + "=" * 70)
    print("VOLATILITY RANKING (normalized ATR/Close, trailing 672 bars)")
    print("=" * 70)
    for seg_name, start, end in SEGMENTS:
        ranked = rank_assets_by_vol(assets_data, start)
        top = [a for a, _ in ranked[:TOP_N]]
        top_per_seg[seg_name] = top
        print(f"\n{seg_name}:")
        for asset, score in ranked:
            mark = "  <-- selected" if asset in top else ""
            print(f"  {asset:<12} {score:>8.4f}%{mark}")

    # -------------------------------------------- run all engines per slice
    asset_results: Dict[str, Dict] = {}
    for asset, df in assets_data.items():
        asset_results[asset] = {}
        for seg_name, start, end in SEGMENTS:
            seg_df = df.iloc[start:end].copy()
            asset_results[asset][seg_name] = {
                "asym": run_asymmetric(seg_df),
                "bin": run_binary(seg_df),
                "tri": run_tristate(seg_df),
            }

    # -------------------------------------------- build portfolio per slice
    seg_port_series: Dict[str, pd.Series] = {}
    seg_port_summary: Dict[str, Dict] = {}
    for seg_name, start, end in SEGMENTS:
        top = top_per_seg[seg_name]
        series_list = [asset_results[a][seg_name]["asym"]["equity"]
                       for a in top]
        port_eq = combine_equity_growth(series_list)
        trades_all: List[Dict] = []
        for a in top:
            trades_all.extend(asset_results[a][seg_name]["asym"]["trades"])
        seg_port_series[seg_name] = port_eq
        seg_port_summary[seg_name] = summarize(port_eq, INITIAL_CAPITAL, trades_all)

    # -------------------------------------------- baseline baskets per slice
    seg_bin_series: Dict[str, pd.Series] = {}
    seg_tri_series: Dict[str, pd.Series] = {}
    for seg_name, start, end in SEGMENTS:
        bin_growth = {a: asset_results[a][seg_name]["bin"]["growth"]
                      for a in assets_data}
        tri_growth = {a: asset_results[a][seg_name]["tri"]["growth"]
                      for a in assets_data}
        seg_bin_series[seg_name] = basket_equity(bin_growth)
        seg_tri_series[seg_name] = basket_equity(tri_growth)

    # -------------------------------------------- consolidated comparison
    print("\n" + "=" * 98)
    print("VALIDATION (untouched OOS) - TOP-3 VOL PORTFOLIO vs INDIVIDUAL ASSETS")
    print("=" * 98)
    hdr = (f"{'Asset/Portfolio':<18}{'ValNet%':>10}{'ValMaxDD%':>11}"
           f"{'ValTrades':>10}{'Win%':>8}{'PF':>8}{'AvgTrd%':>11}{'ValBH%':>9}")
    print(hdr)
    print("-" * len(hdr))

    def _row(name, summ, bh_ret):
        avg = summ["avg_trade_pct"]
        flag = ""
        if summ["trades"] > 0 and avg < FRICTION_RT_PCT:
            flag = "  <-- AVG BELOW FRICTION"
        print(f"{name:<18}{summ['net_return']:>+10.2f}{summ['max_drawdown']:>+11.2f}"
              f"{summ['trades']:>10}{summ['win_rate']:>8.1f}"
              f"{fmt_pf(summ['profit_factor']):>8}{avg:>+11.3f}"
              f"{bh_ret:>+9.2f}{flag}")

    _row("TOP-3 VOL (asym)", seg_port_summary["Validation"], float("nan"))

    ind_rows = []
    for asset in assets_data:
        out = asset_results[asset]["Validation"]["asym"]
        summ = summarize(out["equity"], INITIAL_CAPITAL, out["trades"])
        df = assets_data[asset]
        val_df = df.iloc[TEST_END:]
        bh_ret = (val_df["close"].iloc[-1] / val_df["close"].iloc[0] - 1.0) * 100.0
        ind_rows.append((asset, summ, bh_ret))
    ind_rows.sort(key=lambda x: x[1]["net_return"], reverse=True)
    for asset, summ, bh_ret in ind_rows:
        _row(asset, summ, bh_ret)
    print("-" * len(hdr))
    print(f"Friction round-trip per trade: {FRICTION_RT_PCT:.2f}%")

    # -------------------------------------------- basket vs previous systems
    val_bin_trades: List[Dict] = []
    val_tri_trades: List[Dict] = []
    for a in assets_data:
        val_bin_trades.extend(asset_results[a]["Validation"]["bin"]["trades"])
        val_tri_trades.extend(asset_results[a]["Validation"]["tri"]["trades"])
    val_bin_summ = summarize(seg_bin_series["Validation"], INITIAL_CAPITAL,
                             val_bin_trades)
    val_tri_summ = summarize(seg_tri_series["Validation"], INITIAL_CAPITAL,
                             val_tri_trades)

    print("\n" + "=" * 66)
    print("VALIDATION (untouched OOS) - ASYMMETRIC PORTFOLIO vs PREVIOUS BASKETS")
    print("=" * 66)
    print(f"{'Strategy basket':<24}{'ValNet%':>10}{'ValMaxDD%':>12}{'AvgTrd%':>11}")
    print("-" * 66)
    rows_b = [
        ("Top-3 Vol (asym)", seg_port_summary["Validation"]),
        ("Binary basket (prev)", val_bin_summ),
        ("Tri-State basket (prev)", val_tri_summ),
    ]
    for name, summ in rows_b:
        print(f"{name:<24}{summ['net_return']:>+10.2f}"
              f"{summ['max_drawdown']:>+12.2f}{summ['avg_trade_pct']:>+11.3f}")
    print("-" * 66)

    # -------------------------------------------- friction warning
    print("\nFRICTION SANITY CHECK (avg/trade must beat 0.14% round-trip)")
    print("-" * 60)
    trap = 0
    for item in [("TOP-3 VOL portfolio", seg_port_summary["Validation"])] + [(asset, summ) for asset, summ, _ in ind_rows]:
        name, summ = item
        avg = summ["avg_trade_pct"]
        if summ["trades"] > 0 and avg < FRICTION_RT_PCT:
            trap += 1
        print(f"  {name:<18} avg/trade {avg:+.3f}%")
    print("-" * 60)
    print(f"  {trap}/{len(ind_rows) + 1} names below friction threshold")

    # ---------------------------------------------------------- visualization
    if plt is None:
        print("\nmatplotlib unavailable - skipping chart.")
        return 0

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)

    for ax, seg_name in zip(axes, SEGMENT_NAMES):
        top = top_per_seg[seg_name]

        for asset in top:
            g = asset_results[asset][seg_name]["asym"]["growth"] * INITIAL_CAPITAL
            ax.plot(g.index, g, linewidth=1.0, alpha=0.35, label=f"{asset} asym")

        ax.plot(seg_port_series[seg_name].index, seg_port_series[seg_name],
                label="Top-3 Vol portfolio", color="tab:blue", linewidth=2.2)
        ax.plot(seg_bin_series[seg_name].index, seg_bin_series[seg_name],
                label="Binary basket", color="tab:orange", linewidth=1.8,
                alpha=0.9)
        ax.plot(seg_tri_series[seg_name].index, seg_tri_series[seg_name],
                label="Tri-State basket", color="tab:green", linewidth=1.8,
                alpha=0.9)

        ax.axhline(INITIAL_CAPITAL, color="black", linewidth=0.7, linestyle="--")
        tag = "(Out-of-Sample)" if seg_name == "Validation" else (
            "(In-Sample)" if seg_name == "Training" else "")
        ax.set_title(f"{seg_name} {tag}".strip())
        ax.set_ylabel("Normalized equity (100 = start)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="upper left")

    fig.savefig(OUT_PNG, dpi=110)
    print(f"\nChart saved -> {OUT_PNG}")

    return 0


if __name__ == "__main__":
    sys.exit(main())