"""Multi-asset scalping batch backtester.

Strategy : EMA-200 trend filter + Stochastic RSI entries + ATR stop/target.
Frictions: 0.05% taker fee each side + 0.02% slippage each side
           => ~0.14% round-trip drag per completed trade.

Batches the same rules across 10 liquid Binance USD-M perpetuals so the
evaluation answers "is there a generalized predictive edge after costs?"
instead of "did one asset get lucky?" (Aronson, Evidence-Based Technical
Analysis).

Usage:  python scratch/scalping_batch_backtester.py
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
logger = logging.getLogger("scalping_batch")

# --------------------------------------------------------------------- cfg
TIMEFRAME = "15m"           # toggle to "5m" for the faster sweep
TARGET_CANDLES = 10000      # desired bars per asset
CANDLES = TARGET_CANDLES
INITIAL_CAPITAL = 100.0     # per-asset allocation (USDT)
PAGE_LIMIT = 1000           # Binance USD-M hard cap per fetch_ohlcv call
PAGE_SLEEP = 0.5            # seconds between pagination requests

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
OUT_PNG = os.path.join(OUT_DIR, "scalping_complex_results_10k.png")


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


def stoch_rsi(close: pd.Series, rsi_period: int, stoch_period: int,
              k_smooth: int, d_smooth: int):
    """Return (K, D) Stochastic RSI lines (0-100 scale)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.fillna(50.0)

    lowest = rsi.rolling(stoch_period).min()
    highest = rsi.rolling(stoch_period).max()
    denom = (highest - lowest).replace(0.0, np.nan)
    stoch = 100.0 * (rsi - lowest) / denom
    stoch = stoch.fillna(0.0)

    k = stoch.rolling(k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(d_smooth, min_periods=d_smooth).mean()
    return k, d


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
    """Bar-by-bar simulation. Signals at close t execute at open t+1."""
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
        "signals": int((entry_signal != 0).sum()),
    }


def summarize(symbol: str, trades: List[Dict],
              equity_series: pd.Series) -> Dict:
    final_equity = float(equity_series.iloc[-1])
    net_return = (final_equity / INITIAL_CAPITAL - 1.0) * 100.0
    n = len(trades)

    if n == 0:
        return {
            "symbol": symbol, "final_equity": final_equity,
            "net_return": net_return, "trades": 0, "win_rate": 0.0,
            "profit_factor": 0.0, "avg_trade_pct": 0.0,
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
        "symbol": symbol, "final_equity": final_equity,
        "net_return": net_return, "trades": n, "win_rate": win_rate,
        "profit_factor": profit_factor, "avg_trade_pct": avg_trade_pct,
    }


def fmt_pf(pf: float) -> str:
    return "inf" if np.isinf(pf) else f"{pf:.2f}"


# ------------------------------------------------------------- main driver
def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    exchange = create_exchange()

    print(f"Scalping batch backtest | {TIMEFRAME} | {CANDLES} candles/asset | "
          f"Taker {TAKER_FEE * 100:.2f}%/side | "
          f"Slippage {SLIPPAGE * 100:.2f}%/side")
    print(f"Per-asset allocation: {INITIAL_CAPITAL:.0f} USDT\n")

    results: List[Dict] = []
    growth_map: Dict[str, pd.Series] = {}
    close_map: Dict[str, pd.Series] = {}

    for idx, symbol in enumerate(SYMBOLS, start=1):
        print(f"[{idx}/{len(SYMBOLS)}] Fetching {symbol} ...")
        try:
            df = fetch_10k_ohlcv(exchange, symbol, TIMEFRAME, TARGET_CANDLES)
        except Exception as exc:
            logger.warning("SKIP %s: %s", symbol, exc)
            continue

        if len(df) < EMA_PERIOD + 50:
            logger.warning("SKIP %s: only %d bars (need %d)",
                           symbol, len(df), EMA_PERIOD + 50)
            continue

        out = run_strategy(df)
        summary = summarize(symbol, out["trades"], out["equity"])
        results.append(summary)
        growth_map[symbol] = out["growth"]
        close_map[symbol] = out["close_norm"]

        print(f"     done: bars={len(df)} trades={summary['trades']} "
              f"net={summary['net_return']:+.2f}% "
              f"avg={summary['avg_trade_pct']:+.3f}%")
        time.sleep(SLEEP_BETWEEN_ASSETS)

    if not results:
        print("No asset produced a backtest result.")
        return 1

    # ----------------------------------------------------------- leaderboard
    hdr = (f"{'Asset':<18}{'NetRet%':>10}{'Trades':>8}{'WinRate%':>10}"
           f"{'PF':>8}{'AvgTrd%':>10}")
    print("\n" + "=" * len(hdr))
    print("LEADERBOARD (per-asset $100 allocation)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results, key=lambda x: x["net_return"], reverse=True):
        print(f"{r['symbol']:<18}{r['net_return']:>10.2f}"
              f"{r['trades']:>8}{r['win_rate']:>10.1f}"
              f"{fmt_pf(r['profit_factor']):>8}{r['avg_trade_pct']:>10.3f}")
    print("-" * len(hdr))

    # --------------------------------------------------- friction sanity check
    print("\nFRICTION SANITY CHECK (critical EBTA test)")
    print("-" * 44)
    trap = 0
    for r in results:
        avg = r["avg_trade_pct"]
        if r["trades"] > 0 and avg < (TAKER_FEE + SLIPPAGE) * 2 * 100:
            flag = "  <-- MONEY-LOSING TRAP (edge < round-trip costs)"
            trap += 1
        else:
            flag = ""
        print(f"  {r['symbol']:<18} avg/trade {avg:+.3f}%{flag}")
    print("-" * 44)
    print(f"  Total friction per trade ~ {(TAKER_FEE + SLIPPAGE) * 2 * 100:.2f}%")
    print(f"  Assets below friction threshold: {trap}/{len(results)}")

    # ---------------------------------------------------------- visualization
    if plt is None:
        print("\nmatplotlib unavailable - skipping chart.")
        return 0

    all_index = None
    for s in list(growth_map.values()) + list(close_map.values()):
        all_index = s.index if all_index is None else all_index.union(s.index)
    all_index = all_index.sort_values()

    aligned_growth = pd.DataFrame({
        sym: g.reindex(all_index).ffill() for sym, g in growth_map.items()})
    aligned_close = pd.DataFrame({
        sym: c.reindex(all_index).ffill() for sym, c in close_map.items()})

    combined_strategy = aligned_growth.mean(axis=1) * INITIAL_CAPITAL
    combined_bh = aligned_close.mean(axis=1) * INITIAL_CAPITAL

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True, constrained_layout=True)

    for sym in growth_map:
        ax1.plot(aligned_growth.index, aligned_growth[sym] * INITIAL_CAPITAL,
                 label=sym, linewidth=1.0, alpha=0.85)
    ax1.axhline(INITIAL_CAPITAL, color="black", linewidth=0.7, linestyle="--")
    ax1.set_title(f"Panel 1 - Per-asset equity curves ({TIMEFRAME})")
    ax1.set_ylabel("Equity (USDT, from $100)")
    ax1.legend(fontsize=8, ncol=2, loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(combined_strategy.index, combined_strategy,
             label="Combined scalping basket", color="tab:blue", linewidth=1.6)
    ax2.plot(combined_bh.index, combined_bh,
             label="Equal-weighted buy & hold", color="tab:red",
             linewidth=1.6, alpha=0.8)
    ax2.axhline(INITIAL_CAPITAL, color="black", linewidth=0.7, linestyle="--")
    ax2.set_title("Panel 2 - Basket strategy vs equal-weighted buy & hold")
    ax2.set_ylabel("Equity (USDT, from $100)")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(alpha=0.3)

    fig.savefig(OUT_PNG, dpi=110)
    print(f"\nChart saved -> {OUT_PNG}")
    print(f"Combined basket final   : {combined_strategy.iloc[-1]:.2f} USDT "
          f"({(combined_strategy.iloc[-1] / INITIAL_CAPITAL - 1) * 100:+.2f}%)")
    print(f"Buy & hold basket final : {combined_bh.iloc[-1]:.2f} USDT "
          f"({(combined_bh.iloc[-1] / INITIAL_CAPITAL - 1) * 100:+.2f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())