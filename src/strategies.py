"""Multi-timeframe strategy for BTC/USDT perpetual futures.

Signal pipeline (zero look-ahead):

1. D1 macro trend : bull (1) iff D1 close > 200-day SMA, else bear (0).
2. 1h/4h breakout: Donchian channel (55 bars, EXCLUDING the current bar).
3. OI confirmation: avg(OI t-3..t-1) > avg(OI t-6..t-4) -> new capital enters,
   not liquidations/short squeezes.
4. Macro alignment: longs ONLY in bull trend, shorts ONLY in bear trend.

The raw intent is written to ``position`` at the close of bar t, then shifted
by one bar into ``signal`` (``signal = position.shift(1)``) so the backtester
executes strictly at the open of bar t+1.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DAY_MS = 86_400_000


class MultiTimeframeStrategy:
    """Generate trend-aligned, OI-confirmed Donchian breakout signals."""

    def __init__(self, sma_period: int = 200,
                 donchian_period: int = 55,
                 oi_lookback: int = 3,
                 atr_period: int = 14) -> None:
        self.sma_period = int(sma_period)
        self.donchian_period = int(donchian_period)
        self.oi_lookback = int(oi_lookback)
        self.atr_period = int(atr_period)

    # --- D1 macro trend -------------------------------------------------
    def daily_trend(self, daily_df: pd.DataFrame) -> pd.Series:
        """D1 trend as a Series keyed by the daily CLOSE timestamp (ms).

        ``trend = 1`` if ``close > SMA200`` else ``0``. Keying by (open + 1 day)
        lets us as-of map each intraday bar to the most recently COMPLETED daily
        bar: a bar during day d uses day d-1's trend (no intraday look-ahead).
        """
        close = daily_df["close"].astype(float)
        sma = close.rolling(self.sma_period, min_periods=self.sma_period).mean()
        trend = (close > sma).astype(int)
        close_ts = daily_df["timestamp"].astype("int64") + DAY_MS
        series = pd.Series(trend.values, index=close_ts, name="trend")
        return series.dropna().sort_index()

    def _map_trend_to_execution(self, execution_df: pd.DataFrame,
                                daily_df: pd.DataFrame) -> pd.Series:
        trend_series = self.daily_trend(daily_df)
        if trend_series.empty:
            raise ValueError(
                "Not enough daily bars to build the 200-day SMA trend filter"
            )

        left = execution_df.reset_index(drop=True).copy()
        left["_orig_index"] = execution_df.index
        left = left.sort_values("timestamp")
        right = trend_series.rename("trend").reset_index().sort_values("timestamp")
        mapped = pd.merge_asof(left, right, on="timestamp", direction="backward")
        mapped = mapped.set_index("_orig_index").sort_index()
        return mapped["trend"].reindex(execution_df.index).fillna(0).astype(int)

    # --- Donchian channel (excludes the current bar) ---------------------
    def _donchian(self, execution_df: pd.DataFrame):
        hi = execution_df["high"].shift(1)
        lo = execution_df["low"].shift(1)
        upper = hi.rolling(self.donchian_period,
                           min_periods=self.donchian_period).max()
        lower = lo.rolling(self.donchian_period,
                           min_periods=self.donchian_period).min()
        return upper, lower

    # --- Average True Range (ATR) -----------------------------------------
    def _atr(self, execution_df: pd.DataFrame) -> pd.Series:
        """True Range rolling mean on the execution timeframe.

        Wilder's RMA is the textbook smoothing; a simple rolling mean over the
        same window is used here for consistency with the rest of the pipeline.
        """
        prev_close = execution_df["close"].shift(1)
        tr1 = execution_df["high"] - execution_df["low"]
        tr2 = (execution_df["high"] - prev_close).abs()
        tr3 = (execution_df["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_period,
                          min_periods=self.atr_period).mean()

    # --- OI confirmation -------------------------------------------------
    def _oi_signal(self, execution_df: pd.DataFrame):
        """Return (oi_rising, oi_active).

        ``oi_rising`` is True when short-window OI exceeds the prior window.
        ``oi_active`` marks bars where both OI rolling windows are fully
        populated (recent 29 days); missing/NaN OI yields ``oi_active=False``
        so the caller can bypass the confirmation filter gracefully.
        """
        oi = execution_df["oi_amount"].astype(float)
        recent = oi.shift(1).rolling(self.oi_lookback).mean()  # t-3..t-1
        prior = oi.shift(1 + self.oi_lookback).rolling(self.oi_lookback).mean()  # t-6..t-4
        rising = (recent > prior).fillna(False)
        active = recent.notna() & prior.notna()
        return rising, active

    # --- master signal generator -----------------------------------------
    def generate_signals(self, execution_df: pd.DataFrame,
                         daily_df: pd.DataFrame) -> pd.DataFrame:
        """Augment the execution frame with indicators, ``position`` and ``signal``.

        ``position`` : desired exposure at the CLOSE of bar t (1 / -1 / 0).
        ``signal``   : ``position.shift(1)`` -> executed at bar t+1's open.
        """
        indicator_cols = [
            "trend", "donchian_upper", "donchian_lower", "oi_rising",
            "atr", "breakout_up", "breakout_down", "position", "signal",
        ]
        if execution_df.empty:
            logger.warning(
                "generate_signals received an empty execution DataFrame; "
                "returning an empty frame with the expected columns."
            )
            return pd.DataFrame(columns=list(execution_df.columns) + indicator_cols)

        df = execution_df.copy()

        trend = self._map_trend_to_execution(df, daily_df)
        upper, lower = self._donchian(df)
        atr = self._atr(df)
        oi_rising, oi_active = self._oi_signal(df)

        # If OI is missing/NaN (bars older than ~29 days), bypass the OI
        # confirmation: rely on D1 trend + Donchian breakout alone.
        # If OI is present and valid, the rising-OI filter stays fully active.
        oi_confirmed = ~oi_active | oi_rising

        close = df["close"].astype(float)
        bull = trend == 1
        bear = trend == 0
        breakout_up = close > upper
        breakout_down = close < lower

        long_entry = bull & breakout_up & oi_confirmed
        short_entry = bear & breakout_down & oi_confirmed

        position = pd.Series(0, index=df.index, dtype="int64")
        position[long_entry] = 1
        position[short_entry] = -1

        df["trend"] = trend
        df["donchian_upper"] = upper
        df["donchian_lower"] = lower
        df["oi_rising"] = oi_rising.astype(bool)
        # ATR computed at the close of bar t; shifted so it is known at the
        # open of bar t+1 (no look-ahead leakage into the entry bar).
        df["atr"] = atr.shift(1)
        df["breakout_up"] = breakout_up
        df["breakout_down"] = breakout_down
        df["position"] = position
        # Execute at the OPEN of the NEXT bar -> shift by one (no look-ahead).
        df["signal"] = position.shift(1).fillna(0).astype("int64")

        n_oi_active = int(oi_active.sum())
        n_oi_bypass = int((~oi_active).sum())
        logger.info(
            "OI confirmation: %d bars active, %d bars bypassed (OI missing/NaN).",
            n_oi_active, n_oi_bypass,
        )
        logger.info("Generated %d active raw-position bars.",
                    int((position != 0).sum()))
        return df