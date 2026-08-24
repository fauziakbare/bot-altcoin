"""Strict 3-Segment Walk-Forward Validation for the scalping complex rule system.

Partitions the 10,000-candle series per asset into:
    Training   (in-sample)      : candles 0 .. 4,999
    Testing    (first OOS)      : candles 5,000 .. 7,499
    Validation (untouched OOS)  : candles 7,500 .. 9,999

The exact same "Complex Rule System" (EMA-200 trend + Donchian-20 breakout +
ADX > 25 + Volume > 1.5x SMA-20, SL 2.5x ATR / TP 5.0x ATR, 0.14% round-trip
friction) is run STANDALONE on each slice. Each slice therefore pays its own
indicator warm-up (first ~200 bars produce no trades) and starts fresh at
$100, which is the most conservative, leakage-free way to ask Aronson's EBTA
question: is the edge predictive, or regime-bound data-mining luck?

Usage:  python scratch/scalping_walk_forward.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Dict, List

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
logger = logging.getLogger("scalping_walk_forward")

# --------------------------------------------------------------------- cfg
TIMEFRAME = "15m"
TARGET_CANDLES = 10000
INITIAL_CAPITAL = 100.0     # per-asset, per-segment allocation (USDT)
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
TAKER_FEE = 0.0005          # 0.05% on entry notional + 0.05% on exit notional
SLIPPAGE = 0.0002           # 0.02% adverse fill on entry + 0.02% on exit
FRICTION_RT_PCT = (TAKER_FEE + SLIPPAGE) * 2 * 100   # 0.14% round-trip

# Strategy parameters (complex rule system: trend + breakout + vol + volume).
EMA_PERIOD = 200
DONCHIAN_PERIOD = 20        # fast scalping breakout channel
ADX_PERIOD = 14
ADX_MIN = 25.0              # only trade strong/active trends
VOL_MA_PERIOD = 20
VOL_MULT = 1.5              # volume must exceed 1.5x its 20-bar average
ATR_PERIOD = 14
ATR_SL = 2.5                # stop distance in ATR units
ATR_TP = 5.0                # target distance in ATR units (1:2.0 R:R)

SLEEP_BETWEEN_ASSETS = 0.75  # seconds; ccxt rate limiter also active

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "out")
OUT_PNG = os.path.join(OUT_DIR, "walk_forward_results.png")

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
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return adx


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
    """Paginate Binance USD-M OHLCV backward until `target_candles` unique bars.

    Each fetch_ohlcv call is hard-capped at 1000 bars. Binance treats ``since``
    as an INCLUSIVE startTime and returns up to 1000 bars FORWARD from it. So a
    subsequent page must start one full page-width BEFORE the current earliest
    bar, otherwise the page would re-return the same oldest candle plus newer
    duplicates and never move backward. We stitch, dedup by timestamp, sort.
    """
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

# --------------------------------------------------------------- backtest
def run_strategy(df: pd.DataFrame) -> Dict:
    """Bar-by-bar simulation on ONE standalone slice. Signals at close t
    execute at open t+1. Equity starts fresh at INITIAL_CAPITAL."""
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]

    ema = close.ewm(span=EMA_PERIOD, adjust=False).mean()
    atr = compute_atr(high, low, close, ATR_PERIOD)
    adx = compute_adx(high, low, close, ADX_PERIOD)
    upper, lower = donchian_bands(high, low, DONCHIAN_PERIOD)
    vol_ma = df["volume"].rolling(VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean()

    # All four conditions must be true to fire an entry.
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
    position = 0           # 0 flat | 1 long | -1 short
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

        # 1) Entry at this bar's open from previous bar's signal.
        if position == 0 and i > 0:
            sig = int(entry_signal.iloc[i - 1])
            a = atr.iloc[i - 1]
            if sig != 0 and np.isfinite(a) and a > 0:
                if sig == 1:
                    fill = o * (1.0 + SLIPPAGE)     # long pays up
                else:
                    fill = o * (1.0 - SLIPPAGE)     # short fills down
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

        # 2) Exit checks (stop checked first -> conservative on same-bar hit).
        exit_fill = None
        exit_reason = ""
        if position == 1:
            if l <= sl:
                base = min(o, sl)                   # gap through stop = worse
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

    # 3) Force-close any open position at the final close.
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


def summarize(seg_out: Dict) -> Dict:
    final_equity = float(seg_out["equity"].iloc[-1])
    net_return = (final_equity / INITIAL_CAPITAL - 1.0) * 100.0
    trades = seg_out["trades"]
    n = len(trades)

    if n == 0:
        return {
            "final_equity": final_equity, "net_return": net_return,
            "trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "avg_trade_pct": 0.0,
        }

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

    return {
        "final_equity": final_equity, "net_return": net_return,
        "trades": n, "win_rate": win_rate,
        "profit_factor": profit_factor, "avg_trade_pct": avg_trade_pct,
    }


def fmt_pf(pf: float) -> str:
    return "inf" if np.isinf(pf) else f"{pf:.2f}"


# ------------------------------------------------------------- main driver
def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    exchange = create_exchange()

    print(f"Walk-Forward Validation | {TIMEFRAME} | {TARGET_CANDLES} candles/asset")
    print(f"Friction round-trip: {FRICTION_RT_PCT:.2f}% "
          f"(Taker {TAKER_FEE * 100:.2f}%/side + Slippage {SLIPPAGE * 100:.2f}%/side)")
    print(f"Segments: Training 0-{TRAIN_END - 1} | "
          f"Testing {TRAIN_END}-{TEST_END - 1} | "
          f"Validation {TEST_END}-{TARGET_CANDLES - 1}\n")

    rows: List[Dict] = []
    seg_growth: Dict[str, Dict[str, pd.Series]] = {s: {} for s in SEGMENT_NAMES}
    seg_close: Dict[str, Dict[str, pd.Series]] = {s: {} for s in SEGMENT_NAMES}

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

        df = df.iloc[:TARGET_CANDLES]

        summaries: Dict[str, Dict] = {}
        for seg_name, start, end in SEGMENTS:
            seg_df = df.iloc[start:end].copy()
            out = run_strategy(seg_df)
            summaries[seg_name] = summarize(out)
            seg_growth[seg_name][asset] = out["growth"]
            seg_close[seg_name][asset] = out["close_norm"]

        rows.append({
            "asset": asset,
            "symbol": symbol,
            "train": summaries["Training"],
            "test": summaries["Testing"],
            "val": summaries["Validation"],
        })

        tr = summaries["Training"]
        te = summaries["Testing"]
        va = summaries["Validation"]
        print(f"     train={tr['net_return']:+.2f}% "
              f"test={te['net_return']:+.2f}% "
              f"val={va['net_return']:+.2f}% "
              f"(val trades={va['trades']})")
        time.sleep(SLEEP_BETWEEN_ASSETS)

    if not rows:
        print("No assets processed - aborting.", file=sys.stderr)
        return 1

    # ------------------------------------------------ consolidated table
    hdr = (f"{'Asset':<12}{'TrainNet%':>11}{'TestNet%':>11}{'ValNet%':>11}"
           f"{'ValTrades':>11}{'ValWin%':>9}{'ValPF':>8}{'ValAvgTrd%':>12}")
    print("\n" + "=" * len(hdr))
    print("WALK-FORWARD CONSOLIDATED RESULTS (per-asset $100 per segment)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: x["val"]["net_return"], reverse=True):
        va = r["val"]
        avg = va["avg_trade_pct"]
        flag = ""
        if va["trades"] > 0 and avg < FRICTION_RT_PCT:
            flag = "  <-- MONEY-LOSING TRAP"
        print(f"{r['asset']:<12}{r['train']['net_return']:>+11.2f}"
              f"{r['test']['net_return']:>+11.2f}{va['net_return']:>+11.2f}"
              f"{va['trades']:>11}{va['win_rate']:>9.1f}"
              f"{fmt_pf(va['profit_factor']):>8}{avg:>+12.3f}{flag}")
    print("-" * len(hdr))
    print(f"Friction round-trip per trade: {FRICTION_RT_PCT:.2f}%")

    # ------------------------------------------------ validation verdict
    val_returns = [r["val"]["net_return"] for r in rows]
    val_trades = [r["val"]["trades"] for r in rows]
    pos = sum(1 for v in val_returns if v > 0)
    total_trades = sum(val_trades)
    traps = sum(
        1 for r in rows
        if r["val"]["trades"] > 0 and r["val"]["avg_trade_pct"] < FRICTION_RT_PCT
    )
    print(f"\nValidation (untouched OOS) summary: "
          f"{pos}/{len(rows)} assets positive | "
          f"{total_trades} total trades | {traps} money-losing traps "
          f"(avg/trade < {FRICTION_RT_PCT:.2f}%)")

    # ---------------------------------------------------------- visualization
    if plt is None:
        print("\nmatplotlib unavailable - skipping chart.")
        return 0

    fig, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)

    for ax, seg_name in zip(axes, SEGMENT_NAMES):
        idxs = None
        for s in list(seg_growth[seg_name].values()) + list(
                seg_close[seg_name].values()):
            idxs = s.index if idxs is None else idxs.union(s.index)
        idxs = idxs.sort_values()

        aligned_g = pd.DataFrame({
            a: g.reindex(idxs).ffill()
            for a, g in seg_growth[seg_name].items()})
        aligned_c = pd.DataFrame({
            a: c.reindex(idxs).ffill()
            for a, c in seg_close[seg_name].items()})

        for asset in seg_growth[seg_name]:
            ax.plot(aligned_g.index, aligned_g[asset] * 100,
                    label=asset, linewidth=1.0, alpha=0.85)

        if seg_name == "Validation":
            basket = aligned_g.mean(axis=1) * 100
            bh = aligned_c.mean(axis=1) * 100
            ax.plot(basket.index, basket, label="Combined basket",
                    color="tab:blue", linewidth=1.8)
            ax.plot(bh.index, bh, label="Buy & hold (EW)",
                    color="tab:red", linewidth=1.8, alpha=0.85)

        ax.axhline(100, color="black", linewidth=0.7, linestyle="--")
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
